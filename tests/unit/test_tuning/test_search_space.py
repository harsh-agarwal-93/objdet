"""Unit tests for tuning search space."""


from objdet.tuning.search_space import (
    DETECTION_SEARCH_SPACE,
    YOLO_SEARCH_SPACE,
    SearchSpace,
    define_search_space,
)


class MockTrial:
    """Mock Optuna trial for testing."""

    def __init__(self):
        self.suggested = {}

    def suggest_float(self, name, low, high, log=False):
        value = (low + high) / 2
        self.suggested[name] = value
        return value

    def suggest_int(self, name, low, high, step=1):
        value = (low + high) // 2
        self.suggested[name] = value
        return value

    def suggest_categorical(self, name, choices):
        value = choices[0]
        self.suggested[name] = value
        return value


class TestSearchSpace:
    """Tests for SearchSpace class."""

    def test_add_parameter(self):
        """Test adding parameters."""
        space = SearchSpace()
        space.add("lr", "log_uniform", 1e-5, 1e-2)

        assert "lr" in space.params
        assert space.params["lr"][0] == "log_uniform"

    def test_sample_uniform(self):
        """Test sampling uniform parameter."""
        space = SearchSpace(
            params={
                "weight_decay": ("uniform", 0.0, 0.1),
            }
        )

        trial = MockTrial()
        result = space.sample(trial)

        assert "weight_decay" in result
        assert 0.0 <= result["weight_decay"] <= 0.1

    def test_sample_log_uniform(self):
        """Test sampling log uniform parameter."""
        space = SearchSpace(
            params={
                "lr": ("log_uniform", 1e-5, 1e-2),
            }
        )

        trial = MockTrial()
        result = space.sample(trial)

        assert "lr" in result

    def test_sample_int(self):
        """Test sampling integer parameter."""
        space = SearchSpace(
            params={
                "epochs": ("int", 10, 100),
            }
        )

        trial = MockTrial()
        result = space.sample(trial)

        assert "epochs" in result
        assert isinstance(result["epochs"], int)

    def test_sample_categorical(self):
        """Test sampling categorical parameter."""
        space = SearchSpace(
            params={
                "optimizer": ("categorical", ["adam", "sgd"]),
            }
        )

        trial = MockTrial()
        result = space.sample(trial)

        assert result["optimizer"] in ["adam", "sgd"]


class TestDefineSearchSpace:
    """Tests for define_search_space function."""

    def test_define_search_space(self):
        """Test helper function."""
        space = define_search_space(
            lr=("log_uniform", 1e-5, 1e-2),
            batch_size=("categorical", [8, 16, 32]),
        )

        assert isinstance(space, SearchSpace)
        assert "lr" in space.params
        assert "batch_size" in space.params


class TestPresetSearchSpaces:
    """Tests for preset search spaces."""

    def test_detection_preset(self):
        """Test DETECTION_SEARCH_SPACE preset."""
        assert "learning_rate" in DETECTION_SEARCH_SPACE.params
        assert "batch_size" in DETECTION_SEARCH_SPACE.params

    def test_yolo_preset(self):
        """Test YOLO_SEARCH_SPACE preset."""
        assert "learning_rate" in YOLO_SEARCH_SPACE.params
        assert "momentum" in YOLO_SEARCH_SPACE.params
        assert "mosaic" in YOLO_SEARCH_SPACE.params
