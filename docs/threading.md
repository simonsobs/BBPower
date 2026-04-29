# Threading and Parallelism in `BBCompSep`

This page explains how runtime parallelism works for `BBCompSep`, especially
when `sampler: emcee`. The goal is to make it clear:

- which parameters control parallelism
- where those parameters come from
- which values are defaults and which are effective runtime values
- why the defaults are conservative
- how to choose settings on a shared cluster

If you only remember one thing, remember this:

- `nwalkers` is a YAML sampler setting
- `BBPOWER_EMCEE_WORKERS` is a runtime environment setting
- `OMP_NUM_THREADS` and related variables control native math-library threads
- BBPower does **not** blindly use every CPU you request from Slurm

## 1. The Two Layers of Parallelism

There are two different kinds of parallelism in a typical `BBCompSep` `emcee`
run:

1. Walker-level parallelism
   BBPower evaluates multiple walker log-probabilities at the same time.
   This is controlled by `BBPOWER_EMCEE_WORKERS`.

2. Native math-library threading
   NumPy, BLAS, MKL, OpenBLAS, `numexpr`, and similar libraries may use their
   own threads inside a single likelihood evaluation.
   This is controlled by `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
   `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and related variables.

These two layers interact. If both are large at the same time, the job can
oversubscribe the node and slow down instead of speeding up.

## 2. The Main Runtime Parameters

### `nwalkers`

- Where it is set: the YAML config under `BBCompSep`
- What it means: total number of MCMC walkers in the emcee ensemble
- What it affects: both sampling behavior and the useful upper bound on walker
  parallelism

Example:

```yaml
BBCompSep:
  sampler: emcee
  nwalkers: 40
  n_iters: 10000
```

### `SLURM_CPUS_PER_TASK`

- Where it is set: the Slurm submission script via `#SBATCH --cpus-per-task=...`
- What it means: how many CPUs Slurm allocates to the job
- What it affects: BBPower can use it as the default for
  `BBPOWER_EMCEE_WORKERS`

Example:

```bash
#SBATCH --cpus-per-task=32
```

This usually causes Slurm to export:

```bash
SLURM_CPUS_PER_TASK=32
```

### `BBPOWER_EMCEE_WORKERS`

- Where it is set: the shell environment
- What it means: requested number of parallel emcee likelihood evaluations
- What it affects: the size of the emcee worker pool

BBPower reads this at runtime. If it is not set, BBPower falls back to
`SLURM_CPUS_PER_TASK`, and if that is also missing, it falls back to the local
CPU count.

Important: this is a **requested** value, not always the **effective** value.
BBPower may cap it.

### `BBPOWER_EMCEE_POOL`

- Where it is set: the shell environment
- What it means: which backend emcee uses for parallel likelihood evaluation
- Allowed values: `thread`, `serial`, `process`

Meaning:

- `thread`
  Use a thread pool. This is the default and the recommended mode for
  standard `BBCompSep` runs.
- `serial`
  Disable walker parallelism. Useful for debugging.
- `process`
  Use a process pool. This is only safe if the likelihood object is fully
  picklable.

Why `thread` is the default:

- standard `BBCompSep` likelihoods often include `fgbuster` helper objects that
  are not picklable
- process pools can therefore fail even though serial or threaded runs are fine

### `OMP_NUM_THREADS`

- Where it is set: the shell environment
- What it means: maximum number of OpenMP threads used by native numerical code
- What it affects: some NumPy and compiled-library operations

This does **not** directly mean “threads per walker” in a clean isolated sense.
It is a process-wide setting. In practice, it controls how many native threads
compiled numerical code is allowed to use inside the process.

### `OPENBLAS_NUM_THREADS`

- Where it is set: the shell environment
- What it means: maximum number of OpenBLAS threads
- What it affects: NumPy linear algebra if NumPy is linked against OpenBLAS

### `MKL_NUM_THREADS`

- Where it is set: the shell environment
- What it means: maximum number of Intel MKL threads
- What it affects: NumPy/SciPy linear algebra if linked against MKL

### `NUMEXPR_NUM_THREADS`

- Where it is set: the shell environment
- What it means: maximum number of `numexpr` threads
- What it affects: only code paths using `numexpr`

### `VECLIB_MAXIMUM_THREADS` and `BLIS_NUM_THREADS`

- Where they are set: the shell environment
- What they mean: maximum threads for Apple vecLib and BLIS
- What they affect: only systems using those libraries

They are usually harmless to set to `1` on Linux clusters even if they are not
used.

## 3. How the Bash Defaults Work

The common bash pattern in the runner is:

```bash
export BBPOWER_EMCEE_WORKERS="${BBPOWER_EMCEE_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
export BBPOWER_EMCEE_POOL="${BBPOWER_EMCEE_POOL:-thread}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
```

The syntax:

```bash
${VAR:-default}
```

means:

- if `VAR` is already set and non-empty, keep it
- otherwise use `default`

So this line:

```bash
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
```

means:

- if you already exported `OMP_NUM_THREADS=4`, keep `4`
- if you did not set it, default to `1`

This pattern is useful because it gives the script a safe default while still
letting users override it from the shell or from the Slurm submission
environment.

## 4. How BBPower Chooses the Effective emcee Worker Count

BBPower does not use the requested worker count blindly.

For `emcee`, the useful worker count is capped to:

```text
ceil(nwalkers / 2)
```

Reason:

- emcee's default stretch move is a red-blue move
- it updates one half of the ensemble at a time
- so only about half the walkers are being proposed simultaneously

This means there is no benefit in creating a walker pool larger than about half
the walkers for the default move schedule.

### Example: 40 walkers, 32 CPUs requested

Suppose your Slurm job requests:

```bash
#SBATCH --cpus-per-task=32
```

and your runner contains:

```bash
export BBPOWER_EMCEE_WORKERS="${BBPOWER_EMCEE_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
```

Then:

1. Slurm exports `SLURM_CPUS_PER_TASK=32`
2. bash sets `BBPOWER_EMCEE_WORKERS=32`
3. BBPower sees `nwalkers=40`
4. BBPower caps the useful worker count to `ceil(40 / 2) = 20`

So the log can reasonably show both:

```text
BBPOWER_EMCEE_WORKERS=32
Using 20 emcee worker(s) with thread pool
```

These are not contradictory:

- `32` is the requested environment value
- `20` is the effective runtime value

### Example: 40 walkers, 112 CPUs requested

If you request `112` CPUs but still use `40` walkers:

- requested workers may default to `112`
- effective emcee workers still cap at `20`

So a single chain will not use all 112 CPUs through walker parallelism alone.

## 5. Why the Default Native Thread Count Is `1`

The standard safe cluster configuration is:

```bash
export BBPOWER_EMCEE_POOL=thread
export BBPOWER_EMCEE_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
```

Why use `1` here:

- emcee already parallelizes across likelihood evaluations
- if each worker also launches many BLAS/OpenMP threads, the job can
  oversubscribe the node
- oversubscription often hurts performance more than it helps

This default is conservative and robust. It is a good starting point for
production runs.

## 6. Can One Walker Use More Than One Native Thread?

Yes, but with an important caveat.

If you increase `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, or `MKL_NUM_THREADS`,
compiled numerical kernels inside a likelihood evaluation may use more than one
native thread.

However, in the current threaded emcee setup this is **not** a clean isolated
"N threads per walker" contract. These thread limits are process-wide settings.

Operationally, the useful mental model is:

- reduce `BBPOWER_EMCEE_WORKERS`
- increase `OMP_NUM_THREADS` and the matching BLAS thread settings
- benchmark whether the hybrid setup helps

### Example hybrid configurations

For a run with `nwalkers: 40`, reasonable test points are:

```bash
# Pure walker parallelism
export BBPOWER_EMCEE_WORKERS=20
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

```bash
# Hybrid
export BBPOWER_EMCEE_WORKERS=10
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

```bash
# More aggressive hybrid
export BBPOWER_EMCEE_WORKERS=8
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

These do **not** mean that BBPower has a dedicated scheduler guaranteeing
"2 threads per walker" or "4 threads per walker". They are best understood as
hybrid resource limits that may or may not help depending on the likelihood and
the machine.

## 7. Why Not Use a Process Pool for Cleaner Per-Walker Isolation?

In principle, process pools can make "threads per worker process" easier to
reason about.

In practice, standard `BBCompSep` likelihoods often include non-picklable
objects from `fgbuster`, especially around bandpass-integrated SED helpers.
That makes process-based emcee execution fragile or unusable for common
configurations.

That is why BBPower defaults to:

```bash
BBPOWER_EMCEE_POOL=thread
```

Use:

```bash
BBPOWER_EMCEE_POOL=process
```

only if you know your likelihood object is fully picklable.

## 8. Single-Writer Rule for `emcee.npz.h5`

The emcee HDF5 backend is a single-writer file.

That means:

- one `BBCompSep` emcee run may write `output_dir/emcee.npz.h5`
- a second run must not write the same backend concurrently

Safe:

- let one run finish
- resubmit later to resume from the same backend

Unsafe:

- a local run and a Slurm job writing the same `output_dir`
- two Slurm jobs using the same `output_dir`

Concurrent writers can corrupt the HDF5 backend or make the run fail.

## 9. Recommended Starting Points

### Case A: Standard production run

Use this first unless you have benchmark evidence for something better:

```bash
export BBPOWER_EMCEE_POOL=thread
export BBPOWER_EMCEE_WORKERS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
```

### Case B: 32 CPUs available, 40 walkers

Expected behavior:

- requested workers default to `32`
- effective workers cap to `20`
- extra CPUs are not used by emcee walker parallelism

Good interpretation:

- the run is functioning normally
- the extra CPUs are available, but the chosen emcee move does not have enough
  walker-level concurrency to use them

### Case C: You want to try using more of the node without increasing walkers

Try a benchmark sweep such as:

1. `BBPOWER_EMCEE_WORKERS=20`, BLAS threads `=1`
2. `BBPOWER_EMCEE_WORKERS=10`, BLAS threads `=2`
3. `BBPOWER_EMCEE_WORKERS=8`, BLAS threads `=4`
4. `BBPOWER_EMCEE_WORKERS=5`, BLAS threads `=4`

Do not assume the most aggressive threading setup is the fastest.

## 10. Practical FAQ

### Why does the log show `BBPOWER_EMCEE_WORKERS=32` but `Using 20 emcee worker(s)`?

Because `32` is the requested environment value and `20` is the effective
runtime value after the emcee worker cap is applied for `nwalkers=40`.

### If I change `:-1` to `:-4` in the bash script, what happens?

You changed the default value only.

Example:

```bash
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
```

means:

- use `4` if `OMP_NUM_THREADS` was not already set
- otherwise keep the existing value

### If I want to force a value no matter what, what should I write?

Use:

```bash
export OMP_NUM_THREADS=4
```

without the `${...:-...}` pattern.

### Should I request more CPUs than `ceil(nwalkers / 2)`?

It can still make sense for cluster scheduling, convenience, or benchmarking
hybrid threading setups. But you should not expect a single default emcee chain
to use all of those CPUs automatically.

### Should I increase `nwalkers` just to use more CPUs?

Only if that makes sense for the sampling problem. `nwalkers` is first a
sampling choice, not a cluster-utilization knob.

## 11. Where the Logic Lives in the Code

The main runtime logic is in:

- `bbpower/samplers.py`

The docs that summarize this behavior are:

- [setup.md](setup.md)
- [configuration.md](configuration.md)
- [README.md](../README.md)
