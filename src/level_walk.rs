//! The one shared top-down pyramid level walk (issue #138).
//!
//! Every full-level pyramid cascade in this crate has the same skeleton:
//! start from the full-resolution raster, visit level indices in descending
//! order, and between levels apply a per-tile operation (today always the
//! 2×2 box-filter half downscale, [`crate::resize::downscale_half`]) to
//! produce the next level's raster. Before this module existed that skeleton
//! was transcribed at each call site, and the sites had to be kept in
//! lockstep by hand.
//!
//! [`walk_levels_down`] is that skeleton, written once. The pipeline shape
//! is fixed (descending level order, the top level is never stepped, exactly
//! one step between adjacent levels) while everything site-specific is
//! injected through three hooks:
//!
//! * **`enter`** runs first for every level, before any stepping. This is
//!   where the live engine checks cancellation and emits `LevelStarted`. It
//!   returns a guard value that stays alive for the remainder of the level,
//!   which lets a site hold a tracing span across the step and emit phases.
//!   Sites without per-level setup return `Ok(())`.
//! * **`step`** is the tile operation: it consumes the raster of the level
//!   above and produces this level's raster. It is *not* invoked for the
//!   topmost level (that raster is the walk's input). A site wraps its
//!   operation with whatever bookkeeping it needs (stage timing, memory
//!   tracking); the plain form is `|_, prev| Ok(downscale_half(&prev)?)`.
//!   This closure is the extension point the issue #138 roadmap calls the
//!   tile op: a future rotate/sharpen/colour operation slots in here without
//!   touching any walk site.
//! * **`emit`** runs after the raster for the level is in hand: tile
//!   extraction and sink writes for a live run, byte comparison for a
//!   verify run.
//!
//! Callers today:
//!
//! * the monolithic engine's live run (`engine::run_pyramid`),
//! * the monolithic verify walk (`engine::raster_verify`),
//! * the strip-source verify walk (`stream_verify`), and
//! * the streaming engines' monolithic flush
//!   (`streaming::flush_monolithic_levels`, shared by the sequential and
//!   MapReduce drivers).
//!
//! The strip-phase `downscale_half` applications (the per-strip halving in
//! the streaming strip loops and in `streaming::propagate_down`) are strip
//! transforms rather than full-level walks; they intentionally stay outside
//! this helper.

use crate::raster::Raster;

/// Walk `levels_len` pyramid levels from the top (index `levels_len - 1`)
/// down to level 0, carrying a raster that is transformed by `step` between
/// adjacent levels.
///
/// For each level index `i`, in descending order:
///
/// 1. `enter(i)` runs; its return value is held until the level finishes.
/// 2. If `i` is not the topmost index, `step(i, previous_raster)` produces
///    this level's raster.
/// 3. `emit(i, &raster)` runs.
///
/// Returns the final (level 0) raster so callers can release any external
/// accounting tied to it. A `levels_len` of zero performs no visits and
/// returns `top_raster` unchanged.
pub(crate) fn walk_levels_down<E, G, Enter, Step, Emit>(
    top_raster: Raster,
    levels_len: usize,
    mut enter: Enter,
    mut step: Step,
    mut emit: Emit,
) -> Result<Raster, E>
where
    Enter: FnMut(usize) -> Result<G, E>,
    Step: FnMut(usize, Raster) -> Result<Raster, E>,
    Emit: FnMut(usize, &Raster) -> Result<(), E>,
{
    let mut current = top_raster;
    if levels_len == 0 {
        return Ok(current);
    }
    let top_idx = levels_len - 1;
    for level_idx in (0..levels_len).rev() {
        let _level_guard = enter(level_idx)?;
        if level_idx < top_idx {
            current = step(level_idx, current)?;
        }
        emit(level_idx, &current)?;
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::PixelFormat;
    use crate::raster::RasterError;
    use crate::resize;

    fn gradient_raster(w: u32, h: u32) -> Raster {
        let bpp = PixelFormat::Rgb8.bytes_per_pixel();
        let mut data = vec![0u8; w as usize * h as usize * bpp];
        for y in 0..h {
            for x in 0..w {
                let off = (y as usize * w as usize + x as usize) * bpp;
                data[off] = (x % 256) as u8;
                data[off + 1] = (y % 256) as u8;
                data[off + 2] = ((x + y) % 256) as u8;
            }
        }
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    /// The shared walk must reproduce, byte for byte and level for level,
    /// the inline loop every call site used before the extraction:
    ///
    /// ```text
    /// for level_idx in (0..levels_len).rev() {
    ///     if level_idx < top_level { current = downscale_half(&current)?; }
    ///     visit(level_idx, &current);
    /// }
    /// ```
    #[test]
    fn walk_reproduces_prior_per_site_loop_byte_identically() {
        // 317x201 is deliberately odd-sized so div_ceil rounding at every
        // level exercises the same edge behaviour the engines see.
        let levels_len = 6;
        let src = gradient_raster(317, 201);

        // Reference: the prior per-site inline loop, transcribed verbatim.
        let mut reference: Vec<(usize, u32, u32, Vec<u8>)> = Vec::new();
        {
            let top_level = levels_len - 1;
            let mut current = src.clone();
            for level_idx in (0..levels_len).rev() {
                if level_idx < top_level {
                    current = resize::downscale_half(&current).unwrap();
                }
                reference.push((
                    level_idx,
                    current.width(),
                    current.height(),
                    current.data().to_vec(),
                ));
            }
        }

        // Shared walk with the same tile op.
        let mut walked: Vec<(usize, u32, u32, Vec<u8>)> = Vec::new();
        let final_raster = walk_levels_down::<RasterError, _, _, _, _>(
            src,
            levels_len,
            |_| Ok(()),
            |_, prev| resize::downscale_half(&prev),
            |level_idx, raster| {
                walked.push((
                    level_idx,
                    raster.width(),
                    raster.height(),
                    raster.data().to_vec(),
                ));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(reference, walked, "walk diverged from the inline loop");
        // The returned raster is the last (level 0) one.
        let last = reference.last().unwrap();
        assert_eq!(last.0, 0);
        assert_eq!(final_raster.data(), &last.3[..]);
    }

    /// Hook ordering is load-bearing for observer event streams: `enter`
    /// must fire before the step (the live engine emits `LevelStarted` and
    /// checks cancellation there), the step must not fire for the top
    /// level, and levels must be visited strictly top-down.
    #[test]
    fn enter_step_emit_ordering_matches_engine_contract() {
        let levels_len = 4;
        let src = gradient_raster(64, 64);
        let calls: std::cell::RefCell<Vec<String>> = std::cell::RefCell::new(Vec::new());

        walk_levels_down::<RasterError, _, _, _, _>(
            src,
            levels_len,
            |i| {
                calls.borrow_mut().push(format!("enter{i}"));
                Ok(())
            },
            |i, prev| {
                calls.borrow_mut().push(format!("step{i}"));
                resize::downscale_half(&prev)
            },
            |i, _| {
                calls.borrow_mut().push(format!("emit{i}"));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            calls.into_inner(),
            vec![
                "enter3", "emit3", // top level: no step
                "enter2", "step2", "emit2", "enter1", "step1", "emit1", "enter0", "step0", "emit0",
            ]
        );
    }

    /// The guard returned by `enter` must stay alive across the step and
    /// emit phases of its level (the live engine holds a tracing span in
    /// it), and must be dropped before the next level's `enter`.
    #[test]
    fn enter_guard_spans_step_and_emit() {
        use std::cell::RefCell;
        use std::rc::Rc;

        struct Guard {
            level: usize,
            log: Rc<RefCell<Vec<String>>>,
        }
        impl Drop for Guard {
            fn drop(&mut self) {
                self.log.borrow_mut().push(format!("drop{}", self.level));
            }
        }

        let log: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let src = gradient_raster(16, 16);

        walk_levels_down::<RasterError, _, _, _, _>(
            src,
            3,
            |i| {
                log.borrow_mut().push(format!("enter{i}"));
                Ok(Guard {
                    level: i,
                    log: Rc::clone(&log),
                })
            },
            |i, prev| {
                log.borrow_mut().push(format!("step{i}"));
                resize::downscale_half(&prev)
            },
            |i, _| {
                log.borrow_mut().push(format!("emit{i}"));
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(
            log.borrow().as_slice(),
            [
                "enter2", "emit2", "drop2", "enter1", "step1", "emit1", "drop1", "enter0", "step0",
                "emit0", "drop0",
            ]
        );
    }

    /// A failing enter/step/emit stops the walk immediately and propagates
    /// the error, matching the `?` semantics of the prior inline loops.
    #[test]
    fn errors_short_circuit_like_the_inline_loops() {
        let src = gradient_raster(32, 32);
        let mut visited: Vec<usize> = Vec::new();
        let err = walk_levels_down::<String, _, _, _, _>(
            src,
            5,
            |_| Ok(()),
            |i, prev| {
                if i == 2 {
                    Err(format!("step failed at level {i}"))
                } else {
                    resize::downscale_half(&prev).map_err(|e| e.to_string())
                }
            },
            |i, _| {
                visited.push(i);
                Ok(())
            },
        )
        .unwrap_err();
        assert_eq!(err, "step failed at level 2");
        // Levels 4 and 3 emitted; the failure at level 2's step stops the
        // walk before level 2 emits.
        assert_eq!(visited, vec![4, 3]);
    }

    /// Zero levels: no hooks fire and the input raster comes back unchanged.
    #[test]
    fn zero_levels_is_a_no_op() {
        let src = gradient_raster(8, 8);
        let bytes = src.data().to_vec();
        let out = walk_levels_down::<RasterError, _, _, _, _>(
            src,
            0,
            |_| -> Result<(), RasterError> { panic!("enter must not fire") },
            |_, _| panic!("step must not fire"),
            |_, _| panic!("emit must not fire"),
        )
        .unwrap();
        assert_eq!(out.data(), &bytes[..]);
    }
}
