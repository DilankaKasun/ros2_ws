"""Quick portal connectivity test: run robot + operator in same process."""
import asyncio
import os
import time

os.environ["LIVEKIT_URL"] = "ws://localhost:7880"
os.environ["LIVEKIT_API_KEY"] = "devkey"
os.environ["LIVEKIT_API_SECRET"] = "secret"
os.environ["LIVEKIT_ROOM"] = "test-room"

from livekit import api
from datetime import timedelta

SESSION = "test-room"
LK_URL = "ws://localhost:7880"
LK_KEY = "API5bZnX5w8B5nt"
LK_SECRET = "hsSQgBY5biLxBkGiZ8DVfmKriXF4vhF0S4D6gdK142C"


def _mint_token(identity: str) -> str:
    grants = api.VideoGrants(
        room_join=True, room=SESSION,
        can_publish=True, can_subscribe=True, can_update_own_metadata=True,
    )
    return (
        api.AccessToken(LK_KEY, LK_SECRET)
        .with_identity(identity).with_grants(grants)
        .with_ttl(timedelta(hours=1))
    ).to_jwt()


async def test():
    from livekit.portal import Robot, RobotConfig, Operator, OperatorConfig, DType

    cfg = RobotConfig(SESSION)
    cfg.add_video("cam")
    cfg.add_state_typed([("v", DType.F32)])
    cfg.add_action_typed([("v", DType.F32)])
    cfg.set_fps(10)

    robot = Robot(cfg)
    actions = []

    def on_action(action):
        actions.append(action)
        print(f"[test] action: {action.values}")

    robot.on_action(on_action)
    await robot.connect(LK_URL, _mint_token("test-robot"))
    print("[test] robot connected")

    # Operator
    op_cfg = OperatorConfig(SESSION)
    op_cfg.add_video("cam")
    op_cfg.add_state_typed([("v", DType.F32)])
    op_cfg.add_action_typed([("v", DType.F32)])
    op_cfg.set_fps(10)

    op = Operator(op_cfg)
    await op.connect(LK_URL, _mint_token("test-op"))
    await op.set_active_operator(op.local_identity())
    print(f"[test] operator connected as {op.local_identity()}")

    # Send a state frame
    import time
    ts = int(time.time() * 1_000_000)
    robot.send_state({"v": 1.0}, timestamp_us=ts)
    robot.send_video_frame("cam", bytearray(320 * 240 * 3), width=320, height=240, timestamp_us=ts)

    await asyncio.sleep(0.5)

    # Operator sends action
    op.send_action({"v": 0.5})
    await asyncio.sleep(0.3)

    # Operator claims control
    await op.set_active_operator(op.local_identity())
    op.send_action({"v": 0.75})
    await asyncio.sleep(0.3)

    print(f"[test] actions received by robot: {len(actions)}")
    for a in actions:
        print(f"  -> {a.values}")

    await robot.disconnect()
    await op.disconnect()
    robot.close()
    op.close()
    print("[test] done")


if __name__ == "__main__":
    asyncio.run(test())
