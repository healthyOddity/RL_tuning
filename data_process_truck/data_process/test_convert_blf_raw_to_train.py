import unittest

import numpy as np
import pandas as pd

import convert_blf_raw_to_train as converter


class ConvertBlfRawToTrainTest(unittest.TestCase):
    def test_build_interpolated_resamples_and_maps_training_columns(self):
        raw = pd.DataFrame({
            "timestamp_ms": [1000.0, 1010.0, 1020.0, 1030.0, 1040.0],
            "timestamp_s": [0.0, 0.01, 0.02, 0.03, 0.04],
            "controller_enable_source": [0, 3, 3, 3, 0],
            "total_motor_request_torque_nm": [10.0, 20.0, 30.0, 40.0, 50.0],
            "current_gear": [1, 2, 3, 4, 5],
            "adu_target_steer_deg": [1.0, 2.0, 3.0, 4.0, 5.0],
            "vehicle_speed_kmh": [36.0, 36.0, 72.0, 72.0, 72.0],
            "yaw_rate_radps": [0.0, 0.1, 0.2, 0.3, 0.4],
        })

        out, report = converter.build_interpolated(raw, sample_period_ms=20.0)

        self.assertEqual(list(out["timestamp"]), [0.0, 0.02, 0.04])
        self.assertEqual(list(out["controller_enable"]), [0, 1, 0])
        self.assertEqual(list(out["steering_target"]), [1.0, 3.0, 5.0])
        self.assertEqual(list(out["torque_fl"]), [0.0, 0.0, 0.0])
        self.assertEqual(list(out["torque_fr"]), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(out["torque_rl"].to_numpy(), [189.05, 567.15, 266.0])
        np.testing.assert_allclose(out["torque_rr"].to_numpy(), [189.05, 567.15, 266.0])
        np.testing.assert_allclose(
            out["VehicleInfoBDData.BD18F0090B_VDC2_YawRate"].to_numpy(),
            [0.0, 0.2, 0.4],
        )
        self.assertTrue((out["position_enu.x"] == 0.0).all())
        self.assertTrue((out["position_enu.y"] == 0.0).all())
        self.assertTrue((out["vy"] == 0.0).all())
        self.assertIn("position_enu.x", report["placeholder_columns"])

    def test_controller_enable_only_accepts_mode_three_by_default(self):
        raw = pd.DataFrame({
            "timestamp_ms": [1000.0, 1020.0],
            "controller_enable_source": [3, 4],
        })

        out, _ = converter.build_interpolated(raw, sample_period_ms=20.0)

        self.assertEqual(list(out["controller_enable"]), [1, 0])

    def test_ins_wgs84_converts_to_nearest_origin_enu_and_maps_motion_signals(self):
        raw = pd.DataFrame({
            "timestamp_ms": [1000.0, 1020.0],
            "ins_latitude_deg": [39.068178, 39.068178],
            "ins_longitude_deg": [117.064613, 117.064713],
            "ins_heading_deg": [12.0, 13.0],
            "ins_pitch_deg": [1.0, 1.1],
            "ins_roll_deg": [2.0, 2.1],
            "ins_vbx_mps": [3.0, 3.1],
            "ins_vby_mps": [0.4, 0.5],
        })

        out, report = converter.build_interpolated(raw, sample_period_ms=20.0)

        self.assertEqual(report["origin"]["name"], "tianjin")
        self.assertAlmostEqual(out["position_enu.x"].iloc[0], 0.0, places=3)
        self.assertAlmostEqual(out["position_enu.y"].iloc[0], 0.0, places=3)
        self.assertGreater(out["position_enu.x"].iloc[1], 8.0)
        self.assertLess(abs(out["position_enu.y"].iloc[1]), 0.1)
        self.assertEqual(list(out["euler_angles.z"]), [12.0, 13.0])
        self.assertEqual(list(out["vy"]), [0.4, 0.5])
        self.assertEqual(list(out["ins_pitch_deg"]), [1.0, 1.1])
        self.assertEqual(list(out["ins_roll_deg"]), [2.0, 2.1])
        self.assertEqual(list(out["ins_vbx_mps"]), [3.0, 3.1])

    def test_append_supplement_columns_to_unsegmented_train_csv(self):
        train = pd.DataFrame({"Time_s": [0.0, 0.02]})
        interpolated = pd.DataFrame({
            "ins_pitch_deg": [1.0, 1.1],
            "ins_roll_deg": [2.0, 2.1],
            "ins_vbx_mps": [3.0, 3.1],
        })

        result = converter.append_supplement_columns(train, interpolated)

        self.assertEqual(list(result.columns), ["Time_s", "ins_pitch_deg", "ins_roll_deg", "ins_vbx_mps"])
        self.assertEqual(list(result["ins_vbx_mps"]), [3.0, 3.1])

    def test_all_nan_lat_lon_stays_as_position_placeholder(self):
        raw = pd.DataFrame({
            "timestamp_ms": [1000.0, 1020.0],
            "latitude_deg": [np.nan, np.nan],
            "longitude_deg": [np.nan, np.nan],
        })

        out, report = converter.build_interpolated(raw, sample_period_ms=20.0)

        self.assertTrue((out["position_enu.x"] == 0.0).all())
        self.assertTrue((out["position_enu.y"] == 0.0).all())
        self.assertIn("position_enu.x", report["placeholder_columns"])
        self.assertIn("position_enu.y", report["placeholder_columns"])


if __name__ == "__main__":
    unittest.main()
