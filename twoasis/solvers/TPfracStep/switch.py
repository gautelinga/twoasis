"""The CH-NS step chosen by the time step (chns_solve="auto", with dt_split_max).

The split step (the phase field first, advected by the extrapolated velocity, then the
velocity and pressure with the capillary force of the new phase field) costs less per
step than the coupled step (monolithic.py), but its explicit capillary coupling makes it
unstable above a time step, dt_split_max. That limit depends on the problem (the fluids,
sigma, the mesh, the flow) and is given by the user. Both steps are BDF on the same
levels, with the same coefficients and the same error measure (adapt.py), so either can
follow the other: a switch changes neither the levels nor dt.

Each step takes one of the two:

- the split step's dt is capped at dt_split_max;
- split to coupled when the cap holds the split step back (the dt proposed exceeds
  dt_split_max) and the step the error allows (adapt.Controller.wanted, the
  controller's dt before its limits) exceeds Switch.cost_ratio times dt_split_max. The
  coupled step at that dt then costs less per unit time than the split step capped at
  dt_split_max (cost_ratio: the coupled step's cost per step over the split step's);
- coupled to split when the next dt is at most Switch.switch_down times dt_split_max;
- at most one switch in Switch.hold steps.

The split step is stable at dt_split_max only on steps the error test accepts: far
coarser steps (error measures far above 1) can make it diverge below the limit.
Without dt_adapt nothing measures the step, so the step is the split one when dt <=
switch_down * dt_split_max, and dt stays as given.

The phase field's matrix differs between the two (the split step's has the advection by
the extrapolated velocity; the coupled step's has the advection in a block of its own).
A mode's phase-field preconditioner was last built before the other mode's stretch of
steps, so it is rebuilt on the first step after a switch (this matters when it is kept
over several steps, phase_field_preconditioner_steps > 1). The mode is kept in
NS_parameters["chns_mode"], so that a restart continues in it.

LSODA (Petzold, SIAM J. Sci. Stat. Comput. 4, 1983) switches the same way between a
method with a stability limit (Adams) and a dearer one without (BDF): to the dearer one
when the step it allows is several times the limited one, back when the cheap one's step
is at least as large, with a number of steps between switches.
"""

MODES = ("split", "coupled")


class Switch:
    """The step's mode (`mode`: "split" or "coupled"); `next_dt` after each step.
    `on_switch` is called on each switch (e.g. to have a preconditioner rebuilt)."""

    cost_ratio = 2.5    # the coupled step's cost per step over the split step's
    switch_down = 0.9   # coupled to split when the next dt <= switch_down * dt_split_max
    hold = 20           # the fewest steps between two switches

    def __init__(self, dt_split_max, dt, mode=None, on_switch=None):
        if dt_split_max is None or not dt_split_max > 0:
            raise ValueError("chns_solve='auto' needs dt_split_max > 0, the split step's "
                             "largest stable dt (got {!r})".format(dt_split_max))
        if mode is not None and mode not in MODES:
            raise ValueError("chns_mode {!r}: 'split' or 'coupled'".format(mode))
        self.cap = dt_split_max
        self.mode = mode or ("split" if dt <= self.switch_down * self.cap else "coupled")
        self.on_switch = on_switch
        self.since = self.hold              # steps since the last switch: one may come now
        self.switches = 0

    @property
    def coupled(self):
        return self.mode == "coupled"

    def _set(self, mode):
        self.mode = mode
        self.since = 0
        self.switches += 1
        if self.on_switch is not None:
            self.on_switch(mode)

    def next_dt(self, dt_next, wanted=None):
        """The next step's dt, and its mode in `mode`, after a step: dt_next is the dt
        proposed for it (the controller's, or the fixed dt), `wanted` the step the error
        allows (None: not measured)."""
        cap = self.cap
        self.since += 1
        if self.since >= self.hold:
            if (self.mode == "split" and dt_next > cap and wanted is not None
                    and wanted > self.cost_ratio * cap):
                self._set("coupled")
            elif self.mode == "coupled" and dt_next <= self.switch_down * cap:
                self._set("split")
        return min(dt_next, cap) if self.mode == "split" else dt_next
