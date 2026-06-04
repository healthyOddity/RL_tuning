import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_record_files(record_dir):
    records = []
    for p in record_dir.rglob('*'):
        if p.name.startswith('.'):
            continue
        if p.is_file() and (p.suffix == '.record' or '.record.' in p.name):
            records.append(p)
    return sorted(records)


def step1_extract(record_dir, output_root, force=False):
    interpolated_dir = output_root / '0_interpolated'
    interpolated_dir.mkdir(parents=True, exist_ok=True)

    record_files = _find_record_files(record_dir)
    if not record_files:
        logger.error('[Step 1] No .record files found in %s', record_dir)
        return None

    existing_csv = list(interpolated_dir.glob('*_interpolated.csv'))
    if not force and len(existing_csv) >= len(record_files):
        logger.info('[Step 1] Skipped: %d interpolated CSVs already exist for %d record files', len(existing_csv), len(record_files))
        return interpolated_dir

    for rp in record_files:
        logger.info('[Step 1] Processing %s...', rp.name)
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / 'extract_topic_truck_trq.py'),
             str(rp), '--output-dir', str(interpolated_dir)]
        )
        if result.returncode != 0:
            logger.error('[Step 1] Failed on %s', rp.name)
            return None

    done_csv = list(interpolated_dir.glob('*_interpolated.csv'))
    logger.info('[Step 1] Done. %d records -> %d interpolated CSVs', len(record_files), len(done_csv))
    return interpolated_dir


def step2_classify(input_dir, scenes_dir, smooth_window=5, speed_low=8.33, speed_mid=16.67,
                   min_segment_duration=2.0, enable_plot=True):
    scenes_dir.mkdir(parents=True, exist_ok=True)
    logger.info('[Step 2] Running classify_and_split...')
    cmd = [sys.executable, str(SCRIPT_DIR / 'classify_and_split.py'),
           str(input_dir), str(scenes_dir),
           '--smooth-window', str(smooth_window),
           '--speed-low', str(speed_low),
           '--speed-mid', str(speed_mid),
           '--min-segment-duration', str(min_segment_duration)]
    if not enable_plot:
        cmd.append('--no-plot')
    result = subprocess.run(cmd)
    return result.returncode == 0


def step3_csvdata(scenes_dir, train_dir):
    if not scenes_dir.is_dir():
        logger.error('[Step 3] Scenes dir not found: %s', scenes_dir)
        return False

    subdirs = [d for d in scenes_dir.iterdir() if d.is_dir()]
    if not subdirs:
        logger.warning('[Step 3] No scene subdirectories found in %s', scenes_dir)
        return True

    ok = True
    for subdir in sorted(subdirs):
        csv_paths = sorted(subdir.glob('*.csv'))
        if not csv_paths:
            logger.warning('[Step 3] No CSV files in %s, skipping', subdir.name)
            continue

        train_subdir = train_dir / subdir.name
        train_subdir.mkdir(parents=True, exist_ok=True)

        logger.info('[Step 3] Processing %s: %d segment CSVs...', subdir.name, len(csv_paths))
        cmd = [sys.executable, str(SCRIPT_DIR / 'csvdata_new_truck.py'),
               '--no-segment', '-m'] + [str(p) for p in csv_paths] + ['-o', str(train_subdir)]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error('[Step 3] Failed on %s', subdir.name)
            ok = False
            break

    for subdir in sorted(subdirs):
        train_subdir = train_dir / subdir.name
        train_csvs = list(train_subdir.glob('*_train.csv'))
        logger.info('[Step 3] %s: %d train CSVs generated', subdir.name, len(train_csvs))
    return ok


def parse_args():
    parser = argparse.ArgumentParser(description='Orchestrate full data preprocessing pipeline')
    parser.add_argument('record_dir', type=Path, help='Directory containing .record files')
    parser.add_argument('--output-root', type=Path, default=None,
                        help='Root directory for 0_interpolated/, 1_scenes/, 2_train/ (default: record_dir.parent)')
    parser.add_argument('--skip-step1', action='store_true', help='Skip step 1 (extract/interpolate)')
    parser.add_argument('--skip-step2', action='store_true', help='Skip step 2 (classify & split)')
    parser.add_argument('--skip-step3', action='store_true', help='Skip step 3 (csvdata format convert)')
    parser.add_argument('--force-step1', action='store_true', help='Force rerun step 1 even if outputs exist')
    parser.add_argument('--only-step1', action='store_true', help='Only run step 1')
    parser.add_argument('--only-step2', action='store_true', help='Only run step 2')
    parser.add_argument('--only-step3', action='store_true', help='Only run step 3')
    parser.add_argument('--smooth-window', type=int, default=5, help='Label smoothing window (default: 5)')
    parser.add_argument('--speed-low', type=float, default=8.33, help='Low speed upper bound m/s (default: 8.33)')
    parser.add_argument('--speed-mid', type=float, default=16.67, help='Mid speed upper bound m/s (default: 16.67)')
    parser.add_argument('--min-segment-duration', type=float, default=2.0,
                        help='场景段最小时长（秒），短于此的段被忽略（默认 2.0）')
    parser.add_argument('--no-plot', action='store_true', help='Skip trajectory plotting in step 2')
    parser.add_argument('--no-stats', action='store_true', help='Skip statistics report generation')
    return parser.parse_args()


def run_pipeline(args):
    output_root = args.output_root or args.record_dir.parent
    interpolated_dir = output_root / '0_interpolated'
    scenes_dir = output_root / '1_scenes'
    train_dir = output_root / '2_train'

    logger.info('[Pipeline] record_dir=%s, output_root=%s', args.record_dir, output_root)

    only_flags = [args.only_step1, args.only_step2, args.only_step3]
    any_only = any(only_flags)

    t1 = t2 = t3 = -1.0

    if any_only:
        if args.only_step1:
            start = time.time()
            step1_extract(args.record_dir, output_root, force=args.force_step1)
            t1 = time.time() - start
        if args.only_step2:
            start = time.time()
            if not step2_classify(interpolated_dir, scenes_dir,
                                      smooth_window=args.smooth_window,
                                      speed_low=args.speed_low,
                                      speed_mid=args.speed_mid,
                                      min_segment_duration=args.min_segment_duration,
                                      enable_plot=not args.no_plot):
                return 1
            t2 = time.time() - start
        if args.only_step3:
            start = time.time()
            if not step3_csvdata(scenes_dir, train_dir):
                return 1
            t3 = time.time() - start
        return 0

    if not args.skip_step1:
        start = time.time()
        interp = step1_extract(args.record_dir, output_root, force=args.force_step1)
        t1 = time.time() - start
        if interp is None:
            return 1
    else:
        interp = interpolated_dir

    if not args.skip_step2:
        start = time.time()
        if not step2_classify(interp, scenes_dir,
                              smooth_window=args.smooth_window,
                              speed_low=args.speed_low,
                              speed_mid=args.speed_mid,
                              min_segment_duration=args.min_segment_duration,
                              enable_plot=not args.no_plot):
            return 1
        t2 = time.time() - start

    if not args.skip_step3:
        start = time.time()
        if not step3_csvdata(scenes_dir, train_dir):
            return 1
        t3 = time.time() - start

    if not args.no_stats and train_dir.is_dir():
        logger.info('[Pipeline] Generating statistics report...')
        cmd = [sys.executable, str(SCRIPT_DIR / 'generate_statistics.py'),
               str(train_dir), '--step-times', str(t1), str(t2), str(t3)]
        subprocess.run(cmd)

    return 0


if __name__ == '__main__':
    sys.exit(run_pipeline(parse_args()))
