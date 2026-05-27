"""SBX PPO stack — see docs/specs/2026-05-24-sbx-migration-design.md.

The deploy-time :class:`Controller` lives in ``lsy_drone_racing.control.sbx_song``
(top-level so the loader can pick it via ``controller.file = "sbx_song.py"``).
This subpackage provides the actor / critic modules and the actor-only
checkpoint loader.
"""
