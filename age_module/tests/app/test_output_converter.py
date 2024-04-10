import pytest
from age_module.app.output_converter import AdaptPredAPI


class TestAdaptPredAPI:

    def test_client_output_method(self):
        pred = ({"face": None, "x": 10, "y": 20, "w": 100, "h": 100}, 116)
        api = AdaptPredAPI(*pred)
        result = api.client_output()
        assert "face" not in result  # Check if 'face' key is removed
        assert "age" in result  # Check if 'age' key is added
        assert result["age"] == 116  # Check if age is correctly added

    def test_extract_to_dct(self):
        face_obj = {"face": None, "x": 10, "y": 20, "w": 100, "h": 100}
        expected_output = {"x": 10, "y": 20, "w": 100, "h": 100}
        converter = AdaptPredAPI(face_obj, 30)
        converter.extract_to_dct()
        assert converter.face_obj == expected_output

    def test_age_to_dct(self):
        age = 116
        expected_output = {"age": 116}
        assert (
            AdaptPredAPI(
                face_obj={"x": 10, "y": 20, "w": 100, "h": 100}, age=age
            ).age_to_dct()
            == expected_output
        )


if __name__ == "__main__":
    pytest.main()
