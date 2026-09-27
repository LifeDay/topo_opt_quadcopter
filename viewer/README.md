# Remote progress viewer

`viewer.html` shows `latest.stl` from this folder and reloads it when it changes (it polls
with a HEAD request every 5 s and keeps the camera). three.js 0.186.1 is vendored in `vendor/`.

- Open: http://josh-ms-7a34.taile522b0.ts.net:8000/viewer.html (or http://100.101.26.104:8000/viewer.html).
  `?file=other.stl` shows another file in this folder.
- Publish a model: `uv run python -m topo_opt_quadcopter.stl_export <mesh>` (BESO `.vtk` →
  solid elements only; anything else pyvista reads → its surface). From code:
  `stl_export.export_stl(pyvista_mesh)`. Writes are atomic binary STL.
- Server: `python3 -m http.server` as a systemd user service, bound to the Tailscale IP only.

      ln -sf "$PWD/viewer/topo-viewer.service" ~/.config/systemd/user/
      systemctl --user daemon-reload && systemctl --user enable --now topo-viewer
      loginctl enable-linger "$USER"   # start at boot without a login

  If the machine's Tailscale IP changes, edit `--bind` in `topo-viewer.service`.
