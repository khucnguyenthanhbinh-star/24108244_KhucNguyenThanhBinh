"""Kiểm thử môi trường Sokoban hai thùng."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sokoban.env import SokobanTwoBoxesEnv

MAPS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps")
MAP_FILES = ["map_setting1_small.txt", "map_setting2_corridor.txt",
             "map_setting3_rescue.txt", "map_setting4_advanced.txt"]


def make_env(map_file: str = "map_setting1_small.txt") -> SokobanTwoBoxesEnv:
    return SokobanTwoBoxesEnv(map_path=os.path.join(MAPS_DIR, map_file))


def set_state(env: SokobanTwoBoxesEnv, state_id: int) -> None:
    player, boxes, rescue = env.state_encoder.decode_from_id(state_id)
    env.player = player
    env.boxes = list(boxes)
    env.rescue_state = rescue
    env.steps = 0


@pytest.mark.parametrize("map_file", MAP_FILES)
def test_reset_returns_initial_state(map_file):
    env = make_env(map_file)
    first = env.reset()
    second = env.reset()
    assert first == second
    assert env.boxes == env.initial_boxes
    assert env.player == env.initial_player
    assert env.rescue_state == 0


@pytest.mark.parametrize("map_file", MAP_FILES)
def test_step_signature(map_file):
    env = make_env(map_file)
    env.reset()
    state, reward, terminated, truncated, info = env.step(0)
    assert isinstance(state, int)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert set(info) >= {"is_solved", "is_deadlock", "rescue_state", "steps"}
    assert info["steps"] == 1


@pytest.mark.parametrize("map_file", MAP_FILES)
def test_transition_matches_step(map_file):
    """Mô hình transition(s, a) phải cho kết quả giống hệt step(a)."""
    env = make_env(map_file)
    start = env.reset()

    for state in env.state_ids()[:60]:
        for action in range(4):
            env_a = make_env(map_file)
            set_state(env_a, state)
            next_a, reward_a, term_a = env_a.transition(state, action)

            env_b = make_env(map_file)
            set_state(env_b, state)
            next_b, reward_b, term_b, _, _ = env_b.step(action)

            assert next_a == next_b
            assert reward_a == pytest.approx(reward_b)
            assert term_a == term_b


def test_boxes_order_is_canonical():
    env = make_env()
    player, boxes, rescue = env.state_encoder.decode_from_id(env.reset())
    swapped = list(reversed(boxes))
    assert env.state_encoder.encode_to_id(player, boxes, rescue) == \
        env.state_encoder.encode_to_id(player, swapped, rescue)


def test_step_penalty_applied():
    env = make_env()
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert reward == pytest.approx(env.step_penalty)


def test_goal_reward_and_termination():
    env = make_env()
    env.boxes = list(env.goals)
    env.player = env.initial_player
    state, reward, terminated, _, info = env.step(0)
    assert info["is_solved"]
    assert terminated
    assert reward == pytest.approx(env.step_penalty + env.goal_reward)


def test_rescue_tile_activation_and_single_use():
    env = make_env("map_setting3_rescue.txt")
    assert env.rescue_tile is not None
    env.player = env.rescue_tile
    env.boxes = list(env.initial_boxes)
    env.rescue_state = 0
    env.reset()
    env.player = env.rescue_tile
    _, reward, _, _, info = env.step(0)
    assert info["rescue_state"] == 1
    assert reward == pytest.approx(env.step_penalty + env.rescue_bonus)

    env.rescue_state = 1
    env.player = env.rescue_tile
    _, reward2, _, _, info2 = env.step(0)
    assert info2["rescue_state"] == 1
    assert reward2 < env.rescue_bonus


def test_deadlock_detected():
    env = make_env()
    env.deadlock_detector.corner_deadlocks.add((0, 0))
    assert env.deadlock_detector.is_deadlock([(0, 0)])


def test_state_ids_stable_and_restores_env():
    env = make_env()
    env.reset()
    before = (env.player, list(env.boxes), env.rescue_state, env.steps)
    ids = env.state_ids()
    after = (env.player, list(env.boxes), env.rescue_state, env.steps)
    assert before == after
    assert len(ids) == len(set(ids))
    assert len(ids) > 0


def test_to_char_grid_symbols():
    env = make_env()
    grid = env.to_char_grid()
    assert len(grid) == env.height
    assert all(len(row) == env.width for row in grid)
    flat = "".join("".join(row) for row in grid)
    assert "P" in flat
    assert flat.count("G") == 2
    assert flat.count("B") == 2
