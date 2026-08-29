import math


class ArmKinematics:

    def __init__(self, l0=0.320, l1=0.165, l2=0.140, l3=0.090):
        self.L0 = l0
        self.L1 = l1
        self.L2 = l2
        self.L3 = l3

    def forward(self, theta1, theta2, theta3, theta4):
        th1 = math.radians(theta1)
        th2 = math.radians(theta2)
        th3 = math.radians(theta3)
        th4 = math.radians(theta4)

        th23 = th2 + th3
        th234 = th23 + th4

        r = (
            self.L1 * math.sin(th2)
            + self.L2 * math.sin(th23)
            + self.L3 * math.sin(th234)
        )

        z = self.L0 - (
            self.L1 * math.cos(th2)
            + self.L2 * math.cos(th23)
            + self.L3 * math.cos(th234)
        )

        x = r * math.cos(th1)
        y = r * math.sin(th1)

        return x, y, z

    def _solve_plane(self, wr, wz,
                     theta2_min, theta2_max,
                     theta3_min, theta3_max):
        """2-link IK in the arm plane: shoulder/elbow to put the wrist joint
        at (wr, wz). Returns (theta2, theta3) or None."""
        d_sq = wr ** 2 + (wz - self.L0) ** 2
        d = math.sqrt(d_sq)

        if d > self.L1 + self.L2 + 0.001:
            return None
        if d < abs(self.L1 - self.L2) - 0.001:
            return None

        cos_elbow = (d_sq - self.L1 ** 2 - self.L2 ** 2) / (
            2.0 * self.L1 * self.L2
        )
        if cos_elbow < -1.001 or cos_elbow > 1.001:
            return None
        cos_elbow = max(-1.0, min(cos_elbow, 1.0))
        elbow = math.degrees(math.acos(cos_elbow))

        alpha = math.atan2(wr, self.L0 - wz) if d > 1e-6 else 0.0

        # acos only ever returns 0..180, which is the elbow-down branch. The
        # mirrored elbow-up pose reaches the same point and is often the only
        # one inside the joint limits, so both are offered.
        out = []
        for theta3 in (elbow, -elbow):
            if not (theta3_min <= theta3 <= theta3_max):
                continue
            beta = math.atan2(
                self.L2 * math.sin(math.radians(theta3)),
                self.L1 + self.L2 * math.cos(math.radians(theta3)),
            )
            theta2 = math.degrees(alpha - beta)
            # The same shoulder pose a full turn away is equally valid, and
            # may be the representation that falls inside the limits.
            for t2 in (theta2, theta2 - 360.0, theta2 + 360.0):
                if theta2_min <= t2 <= theta2_max:
                    out.append((t2, theta3))
                    break

        return out or None

    def inverse(self, x, y, z,
                theta2_min=0, theta2_max=180,
                theta3_min=0, theta3_max=180,
                theta4_min=0, theta4_max=180,
                theta1_min=-180, theta1_max=180,
                seed=None):
        """Position-only IK: camera tip reaches (x, y, z). The wrist angle
        theta4 (and therefore the camera aim) is NOT constrained — it is
        whatever the phi sweep lands on, which is why naive scanning swings
        the camera around aimlessly. Prefer inverse_aim() for camera work.

        A point can be reached two ways: the base faces it and the arm
        extends outward, or the base faces away and the arm folds back over
        itself with a negative radius. Only checking the first missed poses
        the arm actually holds — its own home pose reaches +x with the base
        at 180 degrees — so both are tried here, and theta1 is checked
        against the base limits rather than returned blind.

        Among the candidates found, the one needing the least total joint
        movement from `seed` is returned, so the arm takes the nearer route
        instead of whichever branch happened to be swept first.
        """
        base = math.degrees(math.atan2(y, x))
        radius = math.sqrt(x ** 2 + y ** 2)

        candidates = []
        # (theta1, signed radius): facing the point, and facing away from it.
        for theta1, r in ((base, radius), (base + 180.0, -radius)):
            # A base angle out of range may still be legal a turn away.
            for t1 in (theta1, theta1 - 360.0, theta1 + 360.0):
                if not (theta1_min <= t1 <= theta1_max):
                    continue

                # Sweep the whole circle. The old -90..270 window silently
                # excluded poses the arm genuinely holds — its home pose puts
                # the last link at 305 degrees.
                for phi in range(-180, 361, 2):
                    phi_rad = math.radians(phi)

                    wr = r - self.L3 * math.sin(phi_rad)
                    wz = z + self.L3 * math.cos(phi_rad)

                    plane = self._solve_plane(
                        wr, wz,
                        theta2_min, theta2_max,
                        theta3_min, theta3_max,
                    )
                    if plane is None:
                        continue

                    for theta2, theta3 in plane:
                        theta4 = phi - theta2 - theta3
                        for t4 in (theta4, theta4 - 360.0, theta4 + 360.0):
                            if theta4_min <= t4 <= theta4_max:
                                candidates.append((t1, theta2, theta3, t4))
                                break
                break

        if not candidates:
            return None

        if seed is None:
            return candidates[0]

        def travel(c):
            return sum(abs(c[i] - seed[i]) for i in range(min(4, len(seed))))

        return min(candidates, key=travel)

    def inverse_aim(self, sx, sy, sz, ax, ay, az,
                    theta2_min=0, theta2_max=180,
                    theta3_min=0, theta3_max=180,
                    theta4_min=0, theta4_max=180,
                    phi_step=1.0):
        """Orientation-aware IK: put the wrist/camera tip at standoff point
        (sx, sy, sz) AND point the camera (last link, angle th234=phi) at the
        aim point (ax, ay, az) in the arm IK frame.

        Unlike inverse(), this sweeps the wrist angle phi and keeps the
        solution whose camera orientation is closest to the line of sight
        from the standoff point to the aim point, while respecting joint
        limits. Returns (theta1, theta2, theta3, theta4) or None if the
        position is unreachable at all.
        """
        r_c = math.sqrt(sx ** 2 + sy ** 2)
        theta1 = math.degrees(math.atan2(sy, sx))
        cos_t = math.cos(math.radians(theta1))
        sin_t = math.sin(math.radians(theta1))

        # Aim point rotated into the arm's plane (so the "r" axis lies along
        # the base bearing of the standoff point).
        r_p = ax * cos_t + ay * sin_t
        z_p = az

        # Desired camera orientation (last-link angle phi): pointing from the
        # camera toward the aim point. Last link direction in (r,z) is
        # (sin phi, -cos phi); we want that parallel to (r_p-r_c, z_p-sz).
        dr = r_p - r_c
        dz = z_p - sz
        phi_des = math.degrees(math.atan2(dr, -dz))

        best = None
        best_err = math.inf

        # Sweep phi around the desired orientation, keeping whichever
        # reachable pose aims the camera closest to the line of sight.
        for phi in range(-90, 271):
            ph = math.radians(phi)
            wr = r_c - self.L3 * math.sin(ph)
            wz = sz + self.L3 * math.cos(ph)

            plane = self._solve_plane(
                wr, wz,
                theta2_min, theta2_max,
                theta3_min, theta3_max,
            )
            if plane is None:
                continue
            theta2, theta3 = plane[0]

            theta4 = phi - theta2 - theta3
            if not (theta4_min <= theta4 <= theta4_max):
                continue

            # Aim error: angle between actual camera direction and the line
            # from the camera to the aim point.
            to_r = r_p - r_c
            to_z = z_p - sz
            norm = math.hypot(to_r, to_z)
            if norm < 1e-6:
                return theta1, theta2, theta3, theta4
            cos_err = (math.sin(ph) * to_r - math.cos(ph) * to_z) / norm
            cos_err = max(-1.0, min(cos_err, 1.0))
            err = math.degrees(math.acos(cos_err))

            if err < best_err:
                best_err = err
                best = (theta1, theta2, theta3, theta4)

        return best

    def aim_error(self, theta1, theta2, theta3, theta4, ax, ay, az):
        """Degrees between the camera's actual pointing direction and the
        line of sight to aim point (ax, ay, az), given a joint solution."""
        th1 = math.radians(theta1)
        phi = math.radians(theta2 + theta3 + theta4)
        x, y, z = self.forward(theta1, theta2, theta3, theta4)
        cos_t, sin_t = math.cos(th1), math.sin(th1)
        r_p = ax * cos_t + ay * sin_t
        r_c = math.sqrt(x ** 2 + y ** 2)
        to_r = r_p - r_c
        to_z = az - z
        norm = math.hypot(to_r, to_z)
        if norm < 1e-6:
            return 0.0
        cos_err = (math.sin(phi) * to_r - math.cos(phi) * to_z) / norm
        return math.degrees(math.acos(max(-1.0, min(cos_err, 1.0))))

    def is_reachable(self, x, y, z):
        return self.inverse(x, y, z) is not None
