### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 8005bf0e-01ff-42da-967e-414840eb3046
md"""
#### Parag Chettri ~ Santiago Berumen
# An Analysis of the Milky Way's Rotation Curve: A Julia Approach
## Training on Gaia DR3, Testing on an Independent Gaia DR2 Sample
"""

# ╔═╡ f535728b-a9ce-437a-9a0a-dd38945d9040
TableOfContents()

# ╔═╡ 823860b3-4b10-4180-96a0-3539bbaa451f
md"""
For anyone who has looked up at the Milky Way from somewhere truly dark it is easy to think of our galaxy as a still band of light. In reality, it is a spinning disk, and the speed at which its stars orbit tells us how much mass is holding them in place. The *rotation curve*, which is the orbital speed plotted against distance from the Galactic Center, is one of the oldest and most important pieces of evidence that the galaxy contains far more mass than we can see.

This notebook uses astrometry (positions, parallaxes and proper motions) and spectroscopic radial velocities from the European Space Agency's *Gaia* mission to measure that curve ourselves. We train a set of increasingly physical models on **Gaia Data Release 3** (Gaia Collaboration, Vallenari et al. 2023) and then check whether those models hold up on an **independent sample of different stars from Gaia Data Release 2** (Gaia Collaboration, Brown et al. 2018). The project began as a computational astrophysics class project on galactic dynamics, and this version is a substantially reworked edition of it.
"""

# ╔═╡ 9f3c700a-a493-4279-a390-9963715dad5d
begin
	using CSV, DataFrames, HTTP, CodecZlib
	using PlutoUI, PlutoTeachingTools
	using Plots, StatsPlots
	using Statistics, Random, LinearAlgebra, Printf
	using LsqFit, SpecialFunctions, NearestNeighbors
end

# ╔═╡ 8fb686c1-09fc-41d9-97ee-cf54c4e61692
md"""
# The Data: Querying Gaia Ourselves

Instead of relying on CSVs that someone downloaded once and forgot about, the notebook now talks to the ESA Gaia Archive directly using its Table Access Protocol (TAP) service. This brings to light one of the easiest ways a data-driven project can quietly go wrong: if you cannot regenerate your inputs, you cannot really check your outputs.

We select stars that sit close to the Galactic plane ($|b| < 1^\circ$) in two narrow longitude windows: one looking **toward** the Galactic Center ($l \approx 0^\circ$) and one looking directly **away** from it ($l \approx 180^\circ$). Along these two sightlines, a star's orbital motion is almost entirely perpendicular to our line of sight, so it shows up in the proper motion $\mu_l$, and a star's distance from us maps almost one-to-one onto its Galactocentric radius. Every star must have a radial velocity, a parallax measured at better than $5\sigma$, and a well-behaved astrometric solution (RUWE $< 1.4$; Lindegren et al. 2021).
"""

# ╔═╡ ed5e2cfa-a0be-4779-9500-a6a5fc875c1d
md"""
## Keeping the Training and Test Sets Independent

Gaia's data releases are not independent surveys; DR3 is built from a longer stretch of the *same* observations of the *same* stars that DR2 used. Although it seems natural to call a DR2 file "new data," most of its stars are already inside any DR3 sample of the same sky, and testing a model on stars it was trained on tells us very little about how well it generalises.

The solution is to split by **star** rather than by **release**. A star goes into the training set only if its DR3 `random_index` is even, and a DR2 star goes into the test set only if *none* of its DR3 counterparts (from the official `gaiadr3.dr2_neighbourhood` crossmatch) has an even `random_index`. The `random_index` column is a random permutation that Gaia provides for exactly this kind of subsampling, so the split behaves like a fair coin flip. (The parity of `source_id` would not work: every Gaia `source_id` is divisible by 8.) As a second, independent safety net, the notebook crossmatches the two downloaded files on position and removes any test star within 1″ of a training star, a check that should come back with zero matches.
"""

# ╔═╡ a941e758-0c6d-46af-8712-ea3dcbcc32b0
md"""
**Maximum stars per release:** $(@bind n_rows PlutoUI.Select([20_000 => "20,000", 50_000 => "50,000", 100_000 => "100,000"]; default = 100_000))

**Force a fresh download from the Gaia Archive:** $(@bind redownload CheckBox(; default = false))
"""

# ╔═╡ 724a3a8a-8300-4fcb-bda5-957837ef06df
aside(tip(md"""The first run takes a few minutes because the archive runs the queries asynchronously. After that, the results are cached in `DataSets/` and load instantly unless you tick the re-download box or change the row limit."""))

# ╔═╡ 9a8fcaf0-6384-490b-bfe0-305d3943a2ec
queries = (train = adql_dr3_train(n_rows), test = adql_dr2_test(n_rows))

# ╔═╡ cf5bd8c7-d1a2-4b25-8211-01a8e9165142
Markdown.parse("""
### The ADQL queries being sent

**Training (Gaia DR3):**
```sql
$(queries.train)
```

**Test (Gaia DR2):**
```sql
$(queries.test)
```
""")

# ╔═╡ 93073b09-122a-48e2-bfbf-56234bde9768
begin
	DATA_DIR = joinpath(@__DIR__, "DataSets")
	train_path = joinpath(DATA_DIR, "gaia_dr3_train.csv")
	test_path  = joinpath(DATA_DIR, "gaia_dr2_test.csv")
	data_log = String[]
	raw_train, raw_test, data_source = load_or_fetch(queries, train_path, test_path, redownload, data_log)
end;

# ╔═╡ 5ad4cacc-cce9-4c30-aa05-85d8c07a7096
begin
	msg = join(data_log, "\n\n")
	if data_source == :legacy
		warning_box(Markdown.parse(msg * "\n\n**Offline fallback in use.** The legacy files were never a random sample, miss the 355°–360° wedge, and after removing overlapping stars leave only a few hundred test stars. Treat every test-set number below as indicative only."))
	else
		Markdown.parse("**Data source:** `$(data_source)`\n\n" * msg)
	end
end

# ╔═╡ 7cc76540-5367-43de-815a-23fa9cf9a965
begin
	test_independent, overlap_report = remove_overlap(raw_train, raw_test; radius_arcsec = 1.0)
	Markdown.parse("""
	### Overlap audit
	| | Stars |
	|:--|--:|
	| Training stars (DR3) | $(nrow(raw_train)) |
	| Test stars downloaded (DR2) | $(overlap_report.n_test_in) |
	| Test stars sharing a `source_id` with training | $(overlap_report.n_same_id) |
	| Test stars within 1″ of a training star (removed) | $(overlap_report.n_pos_match) |
	| **Independent test stars kept** | **$(overlap_report.n_test_out)** |
	""")
end

# ╔═╡ b1bd2968-b25a-49a1-a92f-2b48e3fc39f9
md"""
# The Solar Parameters

Everything we measure is seen from a moving platform, the Sun, so the numbers we assume about the Sun's position and velocity feed directly into every rotation speed we compute. The first version of this notebook used $R_0 = 8.5$ kpc and $\Theta_0 = 220$ km/s, which are the old IAU 1985 standard values. Those numbers were reasonable decades ago, but modern measurements have moved on and it is imperative that we use them.

| Quantity | Value | Source |
|:--|:--|:--|
| Sun–Galactic Center distance $R_0$ | 8.277 kpc | GRAVITY Collaboration (2022) |
| Solar peculiar motion $U_\odot, V_\odot, W_\odot$ | (11.1, 12.24, 7.25) km/s | Schönrich, Binney & Dehnen (2010) |
| Proper motion of Sgr A* along $l$ | −6.411 mas/yr | Reid & Brunthaler (2020) |
| Total solar azimuthal speed $V_{\odot,\rm tot} = 4.74047\, R_0\, \lvert\mu_{\rm Sgr A^*}\rvert$ | ≈ 251.5 km/s | derived |
| Circular speed at the Sun $\Theta_0 = V_{\odot,\rm tot} - V_\odot$ | ≈ 239.3 km/s | derived |
| Escape speed near the Sun | 528 km/s | Deason et al. (2019) |
| Parallax zero-point (DR3 / DR2) | −0.017 / −0.029 mas | Lindegren et al. (2021) / (2018) |

The trick with Sgr A* is worth pausing on. Mark Reid and Andreas Brunthaler, radio astronomers who tracked the supermassive black hole with the VLBA for almost two decades, found that it drifts across the sky at 6.411 mas/yr. Since Sgr A* is essentially at rest at the Galactic Center, that apparent drift is almost entirely a reflection of *our* orbit, which pins down the Sun's total speed without assuming anything about the shape of the rotation curve.
"""

# ╔═╡ b9fbde21-9a8b-485e-9c3d-35197d7b753d
SUN = let R0 = 8.277, μ_sgrA = 6.411
	Vtot = 4.740470463533348 * R0 * μ_sgrA
	(R0 = R0, U = 11.1, V = 12.24, W = 7.25, Vtot = Vtot, Θ0 = Vtot - 12.24,
	 v_esc = 528.0, zp_dr3 = -0.017, zp_dr2 = -0.029)
end

# ╔═╡ e2675a61-2a64-49df-99e2-b05f38207c84
md"""
# Quality Cuts and Galactocentric Kinematics

Before we can plot a rotation curve, every star needs to pass the same quality cuts and then be moved from "how it looks from Earth" to "how it moves around the Galactic Center." We apply the parallax zero-point correction for each release, repeat the sky-window and parallax cuts on the client (so that the legacy fallback is treated identically), and then run the transformation described below. Finally, we keep stars that orbit in the prograde direction ($V_\phi > 0$) and whose total Galactocentric speed is below the local escape speed, which removes the handful of halo stars and bad measurements that would otherwise skew the medians.
"""

# ╔═╡ 31c7f62d-ca7e-4cef-9a27-0d842d79fffd
begin
	train_all = finalize_sample(quality_cuts(raw_train, SUN.zp_dr3), SUN)
	test_all  = finalize_sample(quality_cuts(test_independent, SUN.zp_dr2), SUN)
	Markdown.parse("""
	After quality cuts and kinematics: **$(nrow(train_all))** training stars and **$(nrow(test_all))** independent test stars.
	The median uncertainty on an individual star's orbital speed is **$(round(median(train_all.vphi_error), digits = 1)) km/s** (training) and **$(round(median(test_all.vphi_error), digits = 1)) km/s** (test), from Monte Carlo propagation of the parallax, proper-motion and radial-velocity errors.
	""")
end

# ╔═╡ a4e08e3f-ea98-4789-bb71-88b68e3e6ace
md"""
# More Readable Data
Later on in the notebook we will see multiple plots that can become difficult to read with tens of thousands of points. To make things easier to understand, you can pick one of four random subsets of the training data. Each subset is drawn by shuffling the stars with a fixed seed, so every subset samples the same range of radii and should lead you to the same conclusions, and if you run the notebook with all four subsets, the fitted parameters should agree to within their uncertainties.

In the first version, the subsets were consecutive blocks of the file, and because Gaia returns rows in sky-position order, one subset was almost entirely inner-Galaxy stars. That is exactly the kind of hidden bias a random shuffle is designed to prevent.
"""

# ╔═╡ da3aa2fd-9caf-4e0f-ab1e-ac8430728af1
md"""
Subset Selection: $(@bind subset_pick PlutoUI.Select([0 => "All training stars", 1 => "Subset 1", 2 => "Subset 2", 3 => "Subset 3", 4 => "Subset 4"]; default = 0))
"""

# ╔═╡ a4399939-2c92-4e72-a969-5ae1d99adb44
aside(tip(md"""Select each subset and compare the fitted parameters in the model-comparison table. Differences larger than the quoted uncertainties would point to a problem."""))

# ╔═╡ d36d20a6-1924-4db9-9271-42bc0c74a748
train = pick_subset(train_all, subset_pick; seed = 416);

# ╔═╡ d14f8cd4-fa6c-4957-9cf4-6f0301ca62b1
details(
md"""
A deeper dive into the conversion from (RA, Dec) to Galactic (l, b). Click if you're curious for more details.""",
md"""
### Conversion from (RA, Dec) to Galactic (l, b)

The relationship between equatorial (ICRS) and Galactic coordinates is a fixed rotation. Gaia's documentation defines it with the matrix $\mathbf{A}_G'$ taken from the Hipparcos catalogue (ESA 1997), so that a unit vector on the sky transforms as

$$\mathbf{r}_{\rm Gal} = \mathbf{A}_G' \, \mathbf{r}_{\rm ICRS}, \qquad \mathbf{r}_{\rm ICRS} = \begin{pmatrix} \cos\delta\cos\alpha \\ \cos\delta\sin\alpha \\ \sin\delta \end{pmatrix}.$$

The rotation is built from three angles: the North Galactic Pole at

$\alpha_{GP} = 192.85948^\circ, \quad \delta_{GP} = 27.12825^\circ$

and the position angle of the Galactic Center meridian, $l_\Omega = 32.93192^\circ$. The Galactic latitude follows from the $z$-component of the rotated vector:

$\sin(b) = \sin(\delta) \sin(\delta_{GP}) + \cos(\delta) \cos(\delta_{GP}) \cos(\alpha - \alpha_{GP})$

Where: $\alpha$ is the Right Ascension (RA) of the star in degrees

Where: $\delta$ is the Declination (Dec) of the star in degrees

Where: $\alpha_{GP}$ and $\delta_{GP}$ are the coordinates of the Galactic north pole

Where: $b$ is the Galactic latitude

Gaia already provides $l$ and $b$ in its catalogue, so we use those directly for the sky cut $|b| < 1^\circ$ (reflected in our query as `ABS(b) < 1`). The first version recomputed $b$ with a separate threshold of $2.5^\circ$, which never removed anything. The full matrix *is* still needed, though, because velocities have to be rotated too, and that is the subject of the next section.
""")

# ╔═╡ c84013f0-b8e2-4677-8f42-cf3906aa761d
details(
md"""
A brief explanation of how we applied space velocity here. Click if you're curious for more details.""",
md"""
The tangential velocity $V_{\text{tan}}$ is given by:

$V_{\text{tan}} = 4.74047 \times D \times \mu$

Where $4.74047$ is the number of km/s corresponding to 1 AU/yr, which converts milliarcseconds per year (mas/yr) at a distance in kiloparsecs (kpc) into km/s.

Where $D$ is the distance in kpc, which we take as $D = 1/\varpi$ after correcting the parallax zero-point.

Where $\mu = \sqrt{\mu_{\alpha*}^2 + \mu_\delta^2}$ is the total proper motion in mas/yr. Gaia's `pmra` is already $\mu_{\alpha*} = \mu_\alpha \cos\delta$, so no extra cosine factor is needed.

The space velocity $V_{\text{space}}$ combines the radial velocity $V_{\text{radial}}$ (along the line of sight) and the tangential velocity (perpendicular to the line of sight):

$V_{\text{space}} = \sqrt{V_{\text{radial}}^2 + V_{\text{tan}}^2}$

**Uncertainties.** The distance uncertainty is $\sigma_D = \sigma_\varpi / \varpi^2$. Because the transformation to Galactocentric velocity is non-linear, we propagate errors by Monte Carlo: each star's parallax, proper motions and radial velocity are redrawn 32 times from their Gaussian errors and pushed through the full transformation, and the scatter of the results becomes that star's uncertainty. (We ignore the catalogue correlation between `pmra` and `pmdec`, which is a small effect for these bright, well-measured stars.)

Now there is something still missing. Gaia measures every velocity relative to the Sun, and the Sun is itself orbiting the Galaxy at about 250 km/s. In our frame of reference, the stars in our neighbourhood that orbit alongside us appear to be almost at rest. We must therefore shift the rest frame to the Galactic Center and use the Galactocentric radius, rather than the distance from the Sun, as our independent variable.
""")

# ╔═╡ ff92d8a6-caa4-4b3f-ae58-dec7062dab73
md"""
# Translating our data to a Galactocentric Rest Frame
"""

# ╔═╡ 8da928c0-6daa-4fc8-bdcd-19905055d9e6
md"""
## **Transformation Function**

The function `galactocentric_kinematics` takes a star's catalogue values and returns its Galactocentric radius and velocity components. It is written out step by step (instead of calling a black-box package) so that each piece of physics is visible, and it has been checked against `astropy`'s `Galactocentric` frame to better than 0.01 km/s.

### **1. Velocity in the Equatorial Frame**

Given the right ascension $\alpha$, declination $\delta$, distance $d$ (kpc), proper motions $\mu_{\alpha*}, \mu_\delta$ (mas/yr) and radial velocity $v_r$ (km/s), we build the star's velocity vector from three unit vectors: $\hat{\mathbf{r}}$ along the line of sight, $\hat{\mathbf{p}}$ pointing east, and $\hat{\mathbf{q}}$ pointing north:

$\mathbf{v}_{\rm ICRS} = v_r\,\hat{\mathbf{r}} + 4.74047\, d \left(\mu_{\alpha*}\,\hat{\mathbf{p}} + \mu_\delta\,\hat{\mathbf{q}}\right)$

$\hat{\mathbf{p}} = (-\sin\alpha,\ \cos\alpha,\ 0), \qquad \hat{\mathbf{q}} = (-\sin\delta\cos\alpha,\ -\sin\delta\sin\alpha,\ \cos\delta)$

### **2. Rotation into Galactic Coordinates**

The position $d\,\hat{\mathbf{r}}$ and the velocity are both rotated with $\mathbf{A}_G'$. This gives heliocentric Galactic positions $x, y, z$, with $x$ pointing toward the Galactic Center, $y$ in the direction of rotation and $z$ toward the North Galactic Pole, together with the matching velocity components $U, V, W$.

The first version tried to do this step by setting $\mu_l = \mu_{\alpha*}/\cos b$ and $\mu_b = \mu_\delta$. Although this seems like a fair shortcut, the equatorial and Galactic "north" directions are tilted relative to each other by an angle that changes across the sky (about $60^\circ$ toward the Galactic Center), so that shortcut mixes the two proper-motion components together.

### **3. Shift to the Galactic Center and Correct for Solar Motion**

The Sun sits at $X = -R_0$ in a frame centered on the Galactic Center:

$X = x - R_0, \qquad Y = y, \qquad Z = z$

$v_X = U + U_\odot, \qquad v_Y = V + V_{\odot} + \Theta_0, \qquad v_Z = W + W_\odot$

The cylindrical Galactocentric radius is $R = \sqrt{X^2 + Y^2}$. We use the cylindrical radius rather than the spherical one because rotation curves describe motion within the disk (for $|b| < 1^\circ$ the difference is tiny anyway). We also neglect the Sun's 20.8 pc height above the plane (Bennett & Bovy 2019), which changes the velocities by far less than a km/s.

### **4. Orbital (Azimuthal) Velocity**

Finally, the velocity is projected onto the direction of Galactic rotation:

$V_\phi = \frac{Y\, v_X - X\, v_Y}{R}, \qquad V_R = \frac{X\, v_X + Y\, v_Y}{R}$

As a sanity check, the Sun itself ($X = -R_0$, $Y = 0$) gives $V_\phi = V_\odot + \Theta_0 \approx 251.5$ km/s.

Sources: [Gaia DR3 documentation, §4.1.7](https://gea.esac.esa.int/archive/documentation/GDR3/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html); Binney & Merrifield (1998), *Galactic Astronomy*, Ch. 10.

### **5. Applying the Transformation**

The function is applied to every star with a loop over the data columns inside `add_kinematics!`, which also runs the Monte Carlo error propagation.
"""

# ╔═╡ d3148eaf-58c5-4669-ac47-bc8d855db6f2
md""" # Distribution of the $V_\phi$
"""

# ╔═╡ eda647a4-53ed-4b5a-ae7e-296c2de6f109
begin
	vphi_mean = mean(train.vphi)
	vphi_med = median(train.vphi)
	vphi_std = std(train.vphi)

	histogram(train.vphi; bins = 200, xlabel = "Orbital Velocity V_φ (km/s)", ylabel = "Density",
		title = "Distribution of V_φ (training stars)", label = "Histogram",
		alpha = 0.6, color = :blue, normalize = :pdf, xlims = (0, 400))
	density!(train.vphi; linewidth = 2, color = :black, label = "KDE")
	vline!([SUN.Θ0]; color = :red, ls = :dash, lw = 2, label = "Θ₀ (adopted)")
	annotate!(20, 0.9 * Plots.ylims()[2],
		text("median = $(round(vphi_med, digits = 1))\nmean = $(round(vphi_mean, digits = 1))\nσ = $(round(vphi_std, digits = 1))", :black, 10, :left))
end

# ╔═╡ 23083e31-93a1-4c72-ab8c-1de6644b82b2
md"""
The distribution peaks just below the adopted circular speed and has a longer tail toward low velocities. This asymmetry is real physics, not noise. Stars on more eccentric orbits spend most of their time near apocenter, where they move more slowly, so any stellar population lags slightly behind the true circular speed, an effect called **asymmetric drift** (Binney & Tremaine 2008, §4.8.2). For the relatively young, cold disk stars that dominate this sample the lag is only a few km/s, but it is the reason we use the **median** velocity in each radial bin rather than the mean.
"""

# ╔═╡ 4bf49b31-1ad1-4dfd-9348-c441860aec42
md"""
# Building the Rotation Curve

The scatter plot of every star is informative but noisy: at any radius, stars have a spread of roughly 20–30 km/s in $V_\phi$. To compare models fairly, we collapse the stars into radial bins and take the **median** $V_\phi$ in each bin. The uncertainty on each median comes from bootstrap resampling (re-drawing the stars in that bin with replacement 200 times), and bins with fewer than 20 stars are dropped.

Of course, with thousands of stars per bin the statistical uncertainty becomes tiny, often below 1 km/s. But the truth is that the Milky Way is not a perfect axisymmetric disk: spiral arms and the central bar push stars around by several km/s, and our solar parameters carry their own errors. We therefore add a **systematic floor** $\sigma_{\rm sys}$ in quadrature, so that each bin's total uncertainty is $\sigma_i^2 = \sigma_{{\rm stat},i}^2 + \sigma_{\rm sys}^2$. Eilers et al. (2019) found systematic uncertainties of a few km/s in a similar analysis, so we default to 3 km/s.
"""

# ╔═╡ 5a3b51e1-cf54-42a3-9e77-def601bd0916
md"""
**Radial bin width (kpc):** $(@bind bin_width Slider(0.25:0.25:1.0; default = 0.5, show_value = true))

**Systematic floor σ_sys (km/s):** $(@bind σ_sys Slider(0.0:0.5:10.0; default = 3.0, show_value = true))
"""

# ╔═╡ 96a4de86-2506-4076-be57-20afee55989a
bins_train = bin_rotation_curve(train.R, train.vphi; width = bin_width, nmin = 20)

# ╔═╡ 7bdaabf2-8f4b-4937-a6f0-1401bdccebbb
begin
	scatter(train.R, train.vphi; ms = 1.5, alpha = 0.15, msw = 0, color = :gray,
		label = "Stars", xlabel = "Galactocentric Radius (kpc)", ylabel = "Orbital Velocity (km/s)",
		title = "Binned Rotation Curve (training)", xlims = (0, 20), ylims = (0, 400),
		legendposition = :bottomright)
	scatter!(bins_train.R, bins_train.V; yerror = sqrt.(bins_train.σ_stat .^ 2 .+ σ_sys^2),
		color = :red, ms = 4, label = "Median per bin (±σ_total)")
	vline!([SUN.R0]; ls = :dot, color = :black, label = "Sun (R₀)")
end

# ╔═╡ b36d7545-7ed8-4b03-b8b3-95b8a9e1fdd2
md"""
Two gaps are worth pointing out. There are very few stars inside ~4 kpc, because a $5\sigma$ parallax cut limits us to stars within a few kpc of the Sun. Similarly, the points thin out beyond ~15 kpc for the same reason. This means that the data genuinely constrain the curve only between about 4 and 15 kpc, something we need to keep in mind before we draw conclusions about the bulge or the outer halo.
"""

# ╔═╡ f5a7f3a0-4585-4873-8bbb-2d1d8ad5443d
md"""
# The Keplerian Orbit
"""

# ╔═╡ ada28aba-9c83-4eb7-8a17-d156ca834281
md"""
As we know, Keplerian orbits are a fundamental part of Newtonian physics, and they give us a baseline to test against. If nearly all of the galaxy's mass were concentrated at its center, the way the Sun dominates the Solar System, the orbital speed would fall off with distance as

$V_{\text{Keplerian}}(R) = \sqrt{\frac{GM_{\text{enclosed}}}{R}}$

Where: $V_{\text{Keplerian}}$ is the orbital velocity in km/s

Where: $M_{\text{enclosed}}$ is the mass that the stars orbit, in solar masses

Where: $R$ is the distance from that mass, in kpc, and $G = 4.30091 \times 10^{-6}$ kpc (km/s)² / M☉
"""

# ╔═╡ aabbc3b5-bb04-4d43-b65f-4841d0d45770
md"""
**Adjust X-Axis Scale (View Window)
Max R (kpc):** $(@bind xmax_kep Slider(1:1:30; default = 20, show_value = true))

**Galactic Enclosed Mass / Solar Masses:** $(@bind mass_slider Slider(1e10:1e10:3e11; default = 1e11, show_value = true))
"""

# ╔═╡ efabe12c-ba23-4fc5-9963-084ca7e5f397
begin
	scatter(train.R, train.vphi; ms = 1.5, alpha = 0.15, msw = 0, label = "Stars",
		xlabel = "Galactocentric Radius (kpc)", ylabel = "Orbital Velocity (km/s)",
		title = "The Keplerian Orbital Curve", xlims = (0, xmax_kep), ylims = (0, 500),
		legendposition = :topright)
	r_kep = 0.1:0.1:xmax_kep
	plot!(r_kep, v_kepler(r_kep, [mass_slider / 1e10]); label = "Keplerian (slider)", lw = 2, color = :red)
	plot!(r_kep, v_kepler(r_kep, fits.kepler.param); label = "Keplerian (best fit)", lw = 2, ls = :dash, color = :black)
	scatter!(bins_train.R, bins_train.V; color = :orange, ms = 3, label = "Binned medians")
end

# ╔═╡ 4ff6dae4-dcc1-40ec-b370-269ed0fc818d
Markdown.parse("""
The best-fitting point mass is \$M = $(@sprintf("%.2e", fits.kepler.param[1] * 1e10))\$ M☉, but it is a poor description: its reduced χ² is **$(round(fit_table.chi2_red[1], digits = 1))**. Needless to say, the rotation of the galaxy is not bound by Keplerian parameters. The mass of the Milky Way is spread throughout its disk and halo, so the enclosed mass keeps growing with radius, and the curve stays nearly flat instead of falling.
""")

# ╔═╡ 42c798a2-e34c-48d5-8321-70f5304fe008
md"""
# A Linear Regression Fit
"""

# ╔═╡ 03dfef59-8ee7-4f66-b691-4d5d1ebeea1c
md"""
The simplest description of a nearly flat curve is a straight line. We write it around the Sun's position,

$V(R) = V_0 + \frac{dV}{dR}\,(R - R_0),$

so that $V_0$ is the circular speed at the Sun and $dV/dR$ is the local slope. Writing it this way (instead of using the intercept at $R = 0$, as the first version did) makes both parameters physically meaningful and much less correlated. It also lets us compare directly to Eilers et al. (2019), who measured $V_0 = 229.0 \pm 0.2$ km/s and a gentle decline of $-1.7 \pm 0.1$ km/s/kpc between 5 and 25 kpc, using $R_0 = 8.122$ kpc.
"""

# ╔═╡ 31d76851-80e8-4ace-a243-bf17bcb4ee6b
aside(tip(md"""Slide $V_0$ and the gradient around to center the residuals on 0 and minimise χ². Then compare your answer to the least-squares fit."""))

# ╔═╡ 97a0da3b-4b17-494e-9753-a12cf6f8787e
md"""
**Adjust Velocity at the Sun, V₀ (km/s):** $(@bind v0_lin Slider(180:1:280; default = 220, show_value = true))

**Adjust Velocity Gradient (km/s/kpc):** $(@bind grad_lin Slider(-10:0.1:10; default = 0.0, show_value = true))

**Adjust X-Axis Scale (View Window) Max R (kpc):** $(@bind xmax_lin Slider(1:1:30; default = 20, show_value = true))
"""

# ╔═╡ e627ac13-0bf8-4543-99e2-5d5f4abf83ca
begin
	p_lin_slider = [Float64(v0_lin), Float64(grad_lin)]
	χ²_lin_slider = chi2(v_linear, p_lin_slider, bins_train, σ_sys)
	scatter(train.R, train.vphi; ms = 1.5, alpha = 0.15, msw = 0, label = "Stars",
		xlabel = "Galactocentric Radius (kpc)", ylabel = "Orbital Velocity (km/s)",
		title = "A Linear Fit", xlims = (0, xmax_lin), ylims = (0, 450), legendposition = :bottomright)
	scatter!(bins_train.R, bins_train.V; color = :orange, ms = 3, label = "Binned medians")
	r_lin = 0:0.1:xmax_lin
	plot!(r_lin, v_linear(r_lin, p_lin_slider); color = :red, lw = 2, label = "Slider model")
	plot!(r_lin, v_linear(r_lin, fits.linear.param); color = :black, lw = 2, ls = :dash, label = "Least-squares fit")
	annotate!(1, 420, text("χ² (slider) = $(round(χ²_lin_slider, digits = 1)) for $(nrow(bins_train)) bins", :black, 10, :left))
end

# ╔═╡ 90e663bc-45aa-4395-96c7-08013b0f44fa
plot_residuals(train, bins_train, v_linear, p_lin_slider; xmax = xmax_lin, title = "Residuals: Observed − Linear (slider)")

# ╔═╡ 5449a8c2-3f18-44c5-a8c1-f4b4ac51fd33
let p = fits.linear.param, e = stderror(fits.linear)
	Markdown.parse("""
	**Least-squares linear fit:** \$V_0\$ = $(round(p[1], digits = 1)) ± $(round(e[1], digits = 1)) km/s, slope = $(round(p[2], digits = 2)) ± $(round(e[2], digits = 2)) km/s/kpc, reduced χ² = $(round(fit_table.chi2_red[2], digits = 2)).

	The binned residuals are the real test here. If they curve up and down in a systematic pattern (rising inside ~7 kpc and flattening outside), then a straight line is missing some structure in the data, even if its overall χ² looks acceptable.
	""")
end

# ╔═╡ ff1628b7-9d49-4095-9b1d-63ad0665c723
md""" ### χ² of the Linear Model
Check if you'd like to see how the χ² varies with the two parameters in this distribution. $(@bind show_lin_heat CheckBox(; default = false))
"""

# ╔═╡ d5f074d7-a034-4af7-9404-70afc5ae86b5
if show_lin_heat
	chi2_heatmap(v_linear, bins_train, σ_sys, 200:0.5:260, -6:0.1:4;
		xlabel = "Gradient (km/s/kpc)", ylabel = "V₀ (km/s)", best = fits.linear.param,
		title = "χ² Surface for the Linear Fit")
end

# ╔═╡ b0fefd6e-5d3f-42de-8e89-2068668a94e3
md"""# A Simple Model"""

# ╔═╡ bc07b054-332e-4f48-bffb-f7da596ca249
begin
	scatter(train.R, train.vphi; ms = 2, alpha = 0.1, msw = 0, label = "Stars",
		xlabel = "Galactocentric Radius (kpc)", ylabel = "V_φ (km/s)",
		title = "Stars with the Binned Median Curve", xlims = (0, 20), ylims = (0, 400),
		legendposition = :bottomright)
	plot!(bins_train.R, bins_train.V; lw = 3, color = :red, marker = :circle, ms = 4,
		label = "Binned median (bin = $(bin_width) kpc)")
end

# ╔═╡ e0901014-ef95-4853-83ce-78fba6be3726
md"""
The binned curve gives us a rough idea of a two-parameter function that could be fitted. It rises in the inner galaxy and then flattens out, which is the classic shape of a disk galaxy's rotation curve. One of the simplest approximations for this shape is

$$f(R) = V_{\infty} \left( 1 - e^{-S R} \right)$$

where $V_\infty$ is the speed that the curve flattens to and $S$ (in kpc⁻¹) controls how quickly it gets there. This function has no physical derivation behind it, it is purely descriptive, but it is a useful stepping stone between a straight line and a real mass model.
"""

# ╔═╡ c9ab4f74-5ce4-44d2-a840-4f1e971efca6
aside(tip(md"""Slide the Velocity and Scaling Factor around to get the best residuals (centered on 0) and the smallest χ² value."""))

# ╔═╡ 1e9a355e-ccf2-4a36-a95c-907a7806e729
md"""
**Adjust Asymptotic Velocity V∞ (km/s):** $(@bind v_inf Slider(180:1:270; default = 220, show_value = true))

**Adjust Exp Scaling Factor S (1/kpc):** $(@bind s_factor Slider(0.05:0.05:3; default = 0.5, show_value = true))

**Adjust X-Axis Scale (View Window) Max R (kpc):** $(@bind xmax_simple Slider(1:1:30; default = 20, show_value = true))
"""

# ╔═╡ 177d7ff2-d445-4b53-91a1-f16fd28414de
begin
	p_simple_slider = [Float64(v_inf), Float64(s_factor)]
	χ²_simple_slider = chi2(v_simple, p_simple_slider, bins_train, σ_sys)
	scatter(train.R, train.vphi; ms = 1.5, alpha = 0.15, msw = 0, label = "Stars",
		xlabel = "Galactocentric Radius (kpc)", ylabel = "Orbital Velocity (km/s)",
		title = "Simple Model", xlims = (0, xmax_simple), ylims = (0, 450), legendposition = :bottomright)
	r_simple = 0:0.1:xmax_simple
	plot!(r_simple, v_simple(r_simple, p_simple_slider); color = :black, lw = 2, label = "Slider model")
	plot!(r_simple, v_simple(r_simple, fits.simple.param); color = :blue, lw = 2, ls = :dash, label = "Least-squares fit")
	plot!(bins_train.R, bins_train.V; lw = 2, color = :red, marker = :circle, ms = 2, label = "Binned medians")
	annotate!(1, 420, text("χ² (slider) = $(round(χ²_simple_slider, digits = 1)) for $(nrow(bins_train)) bins", :black, 10, :left))
end

# ╔═╡ 7350f379-fc0a-4440-a729-8fb59e0a9b6d
plot_residuals(train, bins_train, v_simple, p_simple_slider; xmax = xmax_simple, title = "Residuals: Observed − Simple (slider)")

# ╔═╡ 1a723d6c-64d0-41ef-9714-102f6713416b
md"""  ### χ² of the Simple Model
Check if you'd like to see how the χ² varies with the two parameters in this distribution. $(@bind show_simple_heat CheckBox(; default = false))
"""

# ╔═╡ b4ed7185-f35a-4cc9-bdbf-9d18d3c3ef00
if show_simple_heat
	chi2_heatmap(v_simple, bins_train, σ_sys, 200:0.5:260, 0.05:0.025:2.0;
		xlabel = "Scaling factor S (1/kpc)", ylabel = "V∞ (km/s)", best = fits.simple.param,
		title = "χ² Surface for the Simple Model")
end

# ╔═╡ abaa2538-21b2-4744-aff4-3e8fb18e31a2
md"""
# A Complex Model - The Dark Matter-corrected Milky Way Rotation Curve

The total rotation curve of the Milky Way is the quadrature sum of the contributions from each of its mass components, because the gravitational forces (and therefore the squared circular speeds) simply add:

$V_{\text{total}}(R) = \sqrt{V_{\text{bulge}}^2 + V_{\text{disk}}^2 + V_{\text{HI}}^2 + V_{\text{H2}}^2 + V_{\text{halo}}^2}$

The first version followed the same five-component picture (Ueshima 2010), but used ad-hoc formulas with ten free parameters. While ten knobs might sound more realistic, there are complexities in this matter that must be addressed: several of those formulas did not have units of velocity, and our data only cover about 4–15 kpc, which is far too narrow to separate ten parameters. The fit would drift almost anywhere without the χ² noticing. This version uses the standard, physically derived profile for each component (Binney & Tremaine 2008), fixes the pieces our data cannot constrain at literature values from Paul McMillan's widely used Milky Way mass model (McMillan 2017), and fits only **three** parameters.

#### **1. Bulge (Blue Line): Hernquist sphere, fixed**
$V_{\text{bulge}}(R) = \frac{\sqrt{G M_b R}}{R + a}$

with $M_b = 9 \times 10^9$ M☉ (McMillan 2017) and $a = 0.5$ kpc (Hernquist 1990). Our data barely reach the bulge, so it is held fixed.

#### **2. Stellar Disk (Green Line): Freeman exponential disk, fitted**
An infinitely thin disk with surface density $\Sigma(R) = \Sigma_0 e^{-R/R_d}$ has the exact rotation curve (Freeman 1970)

$V_{\text{disk}}^2(R) = 4\pi G \Sigma_0 R_d\, y^2 \left[I_0(y)K_0(y) - I_1(y)K_1(y)\right], \qquad y = \frac{R}{2R_d}$

where $I_n$ and $K_n$ are modified Bessel functions. **$\Sigma_0$ and $R_d$ are free.** The total disk mass is $M_d = 2\pi\Sigma_0 R_d^2$.

#### **3. HI Layer (Yellow Line): Freeman disk, fixed**
Neutral atomic hydrogen, with $M_{\rm HI} = 1.1 \times 10^{10}$ M☉ and $R_d = 7$ kpc (approximating McMillan 2017, without its central hole).

#### **4. H2 Layer (Pink Line): Freeman disk, fixed**
Molecular hydrogen, with $M_{\rm H_2} = 1.2 \times 10^{9}$ M☉ and $R_d = 1.5$ kpc (same approximation). Both gas layers contribute only ~10–30 km/s, so these simplifications barely matter.

#### **5. Dark Halo (Grey Line): NFW profile, fitted**
Cosmological simulations predict dark matter halos with the Navarro–Frenk–White density profile (Navarro, Frenk & White 1996), $\rho(r) = \rho_0 / [x(1+x)^2]$ with $x = r/r_s$, which gives

$V_{\text{halo}}^2(r) = \frac{4\pi G \rho_0 r_s^3}{r}\left[\ln(1+x) - \frac{x}{1+x}\right]$

**$\rho_0$ is free**; the scale radius is fixed at $r_s = 19.6$ kpc (McMillan 2017), since it lies well outside our data.

#### **6. Final Estimation (Black Line)**
$V_{\text{total}}(R)$ is the quadrature sum. Below is a plot of the model with adjustable starting values for the two mass normalisations.
"""

# ╔═╡ d1eba40e-a444-4017-b884-9d3d2eca4b0f
md"""
**Adjust X-Axis Scale (View Window) Max R (kpc):** $(@bind xmax_cplx Slider(1:1:30; default = 20, show_value = true))

**Disk central surface density Σ₀ (M☉/pc²):** $(@bind Σ0_slider Slider(200:20:2000; default = 900, show_value = true))

**Halo density ρ₀ (10⁻³ M☉/pc³):** $(@bind ρ0_slider Slider(1:0.5:30; default = 8.5, show_value = true))
"""

# ╔═╡ 01c86301-14f5-4926-953d-c05f00759479
begin
	p_cplx_slider = [Σ0_slider / 100, 2.5, ρ0_slider / 10]   # internal units: 1e8 M☉/kpc², kpc, 1e7 M☉/kpc³
	plot_components(train, p_cplx_slider; xmax = xmax_cplx, title = "Complex Model (slider values, R_d = 2.5 kpc)")
end

# ╔═╡ 4ad4950d-2089-4109-b710-44ac69e8fa1a
plot_residuals(train, bins_train, v_complex, p_cplx_slider; xmax = xmax_cplx, title = "Residuals: Observed − Complex (slider)")

# ╔═╡ 420cf99f-649d-4a75-990c-dce52d9d52a3
md""" ## A Non-Linear Regression Fit for the Three Free Parameters
"""

# ╔═╡ d8a71c50-d19e-46d9-96ce-57705039cdf0
md""" ## Nonlinear Least Squares Fit to the Galactic Rotation Curve

Here with the complex model, we fit a physical model of galactic rotation to the binned *Gaia* curve. The model predicts the circular velocity $V_{\text{model}}$ as a function of Galactocentric radius $R$ and the parameter vector $\boldsymbol{\theta} = (\Sigma_0, R_d, \rho_0)$.

### Observed Data

$R_i$ is the median Galactocentric radius of bin $i$

$V_{\text{obs}, i}$ is the median orbital velocity of bin $i$

$\sigma_i$ is the total uncertainty of bin $i$ (bootstrap plus systematic floor)

### Optimization Objective

We look for the parameters that minimise the *weighted* squared differences, the true χ²:

$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \sum_{i=1}^{n} \frac{\left( V_{\text{model}}(R_i; \boldsymbol{\theta}) - V_{\text{obs}, i} \right)^2}{\sigma_i^2} \; + \; \frac{(R_d - 2.6)^2}{0.5^2}$

The last term is an optional **Gaussian prior** on the disk scale length from the review by Joss Bland-Hawthorn and Ortwin Gerhard, which compiled decades of star-count measurements into $R_d = 2.6 \pm 0.5$ kpc (Bland-Hawthorn & Gerhard 2016). It exists because disks and halos are famously degenerate in rotation-curve fitting: a heavier disk and a lighter halo can produce almost the same curve. The minimisation uses the Levenberg–Marquardt algorithm from `LsqFit.jl`, with parameter bounds that keep every mass positive.
"""

# ╔═╡ e02ed27e-58c2-42b7-b4d0-fe7ce368ba38
aside(tip(md"""
Starting point (McMillan 2017 thin disk and halo):

	Σ₀ = 896 M☉/pc²
	R_d = 2.5 kpc
	ρ₀ = 0.00854 M☉/pc³
"""))

# ╔═╡ a5e796de-2448-4de6-a577-361d84801f1b
md"""
**Apply the Bland-Hawthorn & Gerhard (2016) prior on R_d:** $(@bind use_Rd_prior CheckBox(; default = true))
"""

# ╔═╡ 63018b7a-b3a5-41a3-b29c-9e6030597191
fits = fit_all_models(bins_train, σ_sys; Rd_prior = use_Rd_prior)

# ╔═╡ 31e58353-aa5d-4623-a02f-344eb26ef5d0
begin
	plot_components(train, fits.complex.param; xmax = 20, title = "Fitted Complex Model")
	scatter!(bins_train.R, bins_train.V; yerror = sqrt.(bins_train.σ_stat .^ 2 .+ σ_sys^2),
		color = :red, ms = 3, label = "Binned medians")
end

# ╔═╡ c7150c60-a74d-4e0f-9fbe-84684e69fe8e
plot_residuals(train, bins_train, v_complex, fits.complex.param; xmax = 20, title = "Residuals: Observed − Complex (fitted)")

# ╔═╡ a9e6ba8f-8f72-4efc-8adf-aca92a666400
begin
	p_c = fits.complex.param
	e_c = stderror(fits.complex)
	Markdown.parse("""
	### Recovered Parameters from Gaia Data
	| Parameter | Fitted value | McMillan (2017) |
	|:--|--:|--:|
	| Disk central surface density Σ₀ | $(round(p_c[1] * 100, digits = 0)) ± $(round(e_c[1] * 100, digits = 0)) M☉/pc² | 896 M☉/pc² |
	| Disk scale length R_d | $(round(p_c[2], digits = 2)) ± $(round(e_c[2], digits = 2)) kpc | 2.50 kpc |
	| Halo density ρ₀ | $(round(p_c[3] * 10, digits = 2)) ± $(round(e_c[3] * 10, digits = 2)) × 10⁻³ M☉/pc³ | 8.54 × 10⁻³ M☉/pc³ |
	| Optimiser converged? | $(fits.complex.converged) | |
	""")
end

# ╔═╡ 93f0caf3-b524-4fa8-b028-cdf96623dcd4
begin
	md"""
	Select which parameter group you'd like to see the history of optimization for: $(@bind history_choice PlutoUI.Select([1 => "Disk parameters (Σ₀, R_d)", 2 => "Halo parameter (ρ₀)"]; default = 1))
	"""
end

# ╔═╡ a0dce3bf-8ef7-464f-b9c6-8b519732a812
begin
	opt_hist = fits.history
	opt_steps = 1:length(opt_hist)
	if history_choice == 1
		plot(opt_steps, [h[1] * 100 for h in opt_hist]; label = "Σ₀ (M☉/pc²)", lw = 2, color = :green,
			xlabel = "Model evaluation", ylabel = "Σ₀ (M☉/pc²)", legend = :topleft,
			title = "Evolution of the Disk Parameters During Optimization")
		plot!(twinx(), opt_steps, [h[2] for h in opt_hist]; label = "R_d (kpc)", lw = 2, color = :blue,
			ylabel = "R_d (kpc)", legend = :topright)
	else
		plot(opt_steps, [h[3] * 10 for h in opt_hist]; label = "ρ₀ (10⁻³ M☉/pc³)", lw = 2, color = :orange,
			xlabel = "Model evaluation", ylabel = "ρ₀ (10⁻³ M☉/pc³)",
			title = "Evolution of the Halo Parameter During Optimization")
	end
end

# ╔═╡ 69cc2f30-ad9a-42d5-8439-fe710a2e8024
md"""
The history includes every call the optimiser made to the model, including the small parameter nudges it uses to estimate derivatives, which is why the lines look jagged before they settle.
"""

# ╔═╡ 40ecc835-4ebd-45c4-94bb-8b3951137baa
md"""
# Comparing the Models Fairly

A model with more parameters will almost always reach a lower χ², so a lower χ² alone does not prove a model is better. The **Akaike** and **Bayesian Information Criteria** add a penalty for each extra parameter $k$:

$\text{AIC} = \chi^2 + 2k, \qquad \text{BIC} = \chi^2 + k \ln n$

where $n$ is the number of bins. Lower is better, and a BIC difference larger than about 10 is generally considered very strong evidence (Kass & Raftery 1995). Every number in this table uses the same bins, the same uncertainties and the same χ² definition.
"""

# ╔═╡ 40a57abc-8b62-461a-890d-4e60a207eaed
fit_table

# ╔═╡ f1dc2f6e-5d16-4d0e-ae52-2556860fc693
Markdown.parse(model_table_md(fit_table))

# ╔═╡ 94a06dad-e077-4ca5-a249-e151b9acab1c
md"""# Testing Data vs Training Data
For our test data, we use the independent Gaia DR2 stars selected above, stars that are **not** in the training set. We apply exactly the same quality cuts, zero-point correction and Galactocentric transformation (the first version skipped the quality cuts on the test set), bin them the same way, and then ask how well each model **trained on DR3** predicts them, without refitting anything.

Because DR2 has fewer radial velocities than DR3, and half of its stars were reserved for training, the test set is smaller and its bins need at least 10 stars instead of 20.
"""

# ╔═╡ 29038359-ce88-45ff-82b3-1863379e1880
md"""
Check if you'd like to see how the test data compares to the training data. $(@bind ready_to_test CheckBox(; default = false))
"""

# ╔═╡ 4722e1b0-e060-4aec-9525-357e2b0ca5fb
bins_test = bin_rotation_curve(test_all.R, test_all.vphi; width = bin_width, nmin = 10);

# ╔═╡ 149f0079-875e-4ce0-a48b-1915d40b0aef
if ready_to_test
	scatter(test_all.R, test_all.vphi; ms = 2, alpha = 0.2, msw = 0, color = :gray,
		label = "Test stars (DR2)", xlabel = "Galactocentric Radius (kpc)",
		ylabel = "Orbital Velocity (km/s)", title = "Models Trained on DR3 vs Independent DR2 Test Data",
		xlims = (0, 20), ylims = (0, 450), legendposition = :bottomright)
	scatter!(bins_test.R, bins_test.V; yerror = sqrt.(bins_test.σ_stat .^ 2 .+ σ_sys^2),
		color = :red, ms = 4, label = "Test medians")
	r_test = 0.2:0.1:20
	plot!(r_test, v_linear(r_test, fits.linear.param); lw = 2, color = :green, label = "Linear (trained)")
	plot!(r_test, v_simple(r_test, fits.simple.param); lw = 2, color = :blue, label = "Simple (trained)")
	plot!(r_test, v_complex(r_test, fits.complex.param); lw = 3, color = :black, label = "Complex (trained)")
end

# ╔═╡ 37806728-128f-40cb-a339-92d4efcea87b
if ready_to_test
	test_table = evaluate_on_test(fits, bins_test, σ_sys)
	Markdown.parse(test_table_md(test_table, nrow(bins_test)))
end

# ╔═╡ a19ee871-c665-4daf-920f-86882a93610e
if ready_to_test
	refit = fit_all_models(bins_test, σ_sys; Rd_prior = use_Rd_prior)
	Markdown.parse(parameter_shift_md(fits, refit))
end

# ╔═╡ 77fd543a-1eb7-4458-b83f-8f631ce3c927
md"""
The second table asks a slightly different question: if we refit each model on the test stars alone, do we recover the same parameters? Agreement within the uncertainties means the trends we found are properties of the Galaxy rather than of one particular sample. A disagreement is still a useful result, and it is exactly the kind of null or negative outcome that should be reported rather than hidden.
"""

# ╔═╡ c51ca757-c00c-4603-b1c4-cdccd0833d2b
md""" # Conclusion
"""

# ╔═╡ 75b540af-e42d-4c72-a616-d182d3a5bfff
Markdown.parse(conclusion_md(fit_table, fits, @isdefined(test_table) ? test_table : nothing, data_source))

# ╔═╡ 20d66522-b432-40a8-a64f-d5e8caac02ce
md"""
### Why did the first version reach the opposite conclusion?

The original notebook reported that the Simple model beat the Complex model, and it offered overfitting as an explanation. Although overfitting is a fair concern in general, it fails to explain what actually happened here. The two models were scored with different formulas, a Pearson-style $\sum(O-E)^2/E$ on binned means for the Simple model, and an unweighted $\sum(O-E)^2$ divided by the degrees of freedom on individual stars for the Complex model, so the two numbers were never comparable. In addition, the ten-parameter model was degenerate, the "test" data were the training data, and the velocity transformation carried errors of ~10–30 km/s. Once those issues are fixed, the comparison becomes a fair one.

### What the results can and cannot tell us

Of course, a better χ² does not mean our mass model is *correct*. The data cover only about 4–15 kpc, and we have ignored asymmetric drift (which makes $V_\phi$ a few km/s lower than the true circular speed), the gravitational influence of the bar and spiral arms, and the vertical structure of the disk. The fixed bulge, gas and halo scale radius also carry their own uncertainties, which are not included in the quoted errors. In essence, the Complex model is the most *physical* description we tested, but its parameters should be read as a consistency check against the literature rather than as a new measurement.
"""

# ╔═╡ 673bfffb-e887-4a1a-b7db-40405c76e6d5
details(
md"""
A deeper dive into the uses of these results in approximating the dark matter content of the Milky Way! Click if you're curious for more details.""",
md"""
### Inferring the Mass Distribution from the Rotation Curve

Given the total rotation curve,

$V_{\text{total}}(R) = \sqrt{V_{\text{bulge}}^2 + V_{\text{disk}}^2 + V_{\text{HI}}^2 + V_{\text{H2}}^2 + V_{\text{halo}}^2}$

we can deduce how the mass of the Milky Way is distributed using Newtonian dynamics.

#### 1. Total Enclosed Mass

For a spherical mass distribution,

$M_{\text{total}}(<R) = \frac{R\, V_{\text{total}}^2(R)}{G}$

gives the **cumulative mass** inside a given radius purely from the rotation speed. For a flattened disk this formula is only approximate, since a disk produces a slightly higher circular speed than a sphere of the same enclosed mass, so we call it the *spherical-equivalent* mass.

#### 2. Component-wise Mass Contributions

For the two spherical components, we use the exact enclosed masses:

- Hernquist bulge: $M_{\text{bulge}}(<r) = M_b \dfrac{r^2}{(r+a)^2}$
- NFW halo: $M_{\text{halo}}(<r) = 4\pi\rho_0 r_s^3 \left[\ln(1+x) - \dfrac{x}{1+x}\right]$

For the three disks, we use the spherical-equivalent $M_i = R V_i^2 / G$.

#### 3. The Dark Matter Fraction

The cleanest dynamical quantity is the fraction of the squared circular speed supplied by the halo,

$f_{\rm DM}(R) = \frac{V_{\text{halo}}^2(R)}{V_{\text{total}}^2(R)},$

which is the fraction of the inward gravitational pull at radius $R$ that comes from dark matter.

#### 4. The Local Dark Matter Density

The NFW profile evaluated at the Sun gives the dark matter density in our own neighbourhood, $\rho_{\rm DM}(R_0) = \rho_0 / [x_0(1+x_0)^2]$. This number sets the expected signal in direct-detection experiments like XMASS, XENONnT and LZ, and it is usually quoted in GeV/cm³ (1 M☉/pc³ ≈ 38.0 GeV/cm³). Justin Read's review of local dark matter measurements places it around 0.2–0.6 GeV/cm³ (Read 2014; see also de Salas & Widmark 2021).

These quantities are derived **entirely from dynamics**, with **no luminosity data required**!
""")

# ╔═╡ ce3c8382-9ec2-42a3-87c4-52d83ba0880c
md""" ### Unveil the Dark Matter Content of the Milky Way
Although this is derived using Newtonian principles and not full general relativity, weak-field Newtonian gravity is an excellent approximation at galactic scales, so it is still quite a significant estimate!
Check to see: $(@bind show_mass CheckBox(; default = false))
"""

# ╔═╡ 7779c5c9-5637-4a0e-9d5e-44fa8b67dff4
if show_mass
	mass_plots(fits.complex.param)
end

# ╔═╡ 101356f0-000b-4f14-9fba-881ef7f0fd0f
if show_mass
	Markdown.parse(mass_summary_md(fits.complex, SUN))
end

# ╔═╡ 09b90ac7-3525-444e-8775-477a47b24b5c
md"""
# Where This Could Go Next

It is not enough to stop at a single fit. The most natural next step is to replace $1/\varpi$ with the probabilistic distances of Coryn Bailer-Jones and collaborators (Bailer-Jones et al. 2021), which would let us relax the $5\sigma$ parallax cut and push the curve toward 20 kpc and beyond. After that, correcting for asymmetric drift with the Jeans equation (as Eilers et al. 2019 did) would turn our stellar $V_\phi$ into a true circular speed, and adding fainter stars from the Gaia DR3 kinematics catalogues would populate the inner galaxy where the bulge and bar live. Perhaps most exciting, *Gaia* DR4 is expected to roughly double the time baseline of the mission, which should shrink the proper-motion uncertainties several-fold. Let's stay tuned, this is a notebook that can simply be re-run the day DR4 lands in the archive, and we believe that the rotation curve of our own galaxy will only come into sharper focus from here.
"""

# ╔═╡ 7fa4dbd3-4898-4dc2-a5a3-5f85569d0061
md"""
# References

Bailer-Jones, C. A. L. (2015). Estimating Distances from Parallaxes. *PASP*, 127, 994. [doi:10.1086/683116](https://doi.org/10.1086/683116)

Bailer-Jones, C. A. L., Rybizki, J., Fouesneau, M., Demleitner, M., & Andrae, R. (2021). Estimating Distances from Parallaxes. V. *AJ*, 161, 147. [doi:10.3847/1538-3881/abd806](https://doi.org/10.3847/1538-3881/abd806)

Bennett, M., & Bovy, J. (2019). Vertical waves in the solar neighbourhood in Gaia DR2. *MNRAS*, 482, 1417. [doi:10.1093/mnras/sty2813](https://doi.org/10.1093/mnras/sty2813)

Binney, J., & Merrifield, M. (1998). *Galactic Astronomy*. Princeton University Press.

Binney, J., & Tremaine, S. (2008). *Galactic Dynamics* (2nd ed.). Princeton University Press.

Bland-Hawthorn, J., & Gerhard, O. (2016). The Galaxy in Context. *ARA&A*, 54, 529. [doi:10.1146/annurev-astro-081915-023441](https://doi.org/10.1146/annurev-astro-081915-023441)

de Salas, P. F., & Widmark, A. (2021). Dark matter local density determination. *Rep. Prog. Phys.*, 84, 104901. [doi:10.1088/1361-6633/ac24e7](https://doi.org/10.1088/1361-6633/ac24e7)

Deason, A. J., Fattahi, A., Belokurov, V., Evans, N. W., Grand, R. J. J., Marinacci, F., & Pakmor, R. (2019). The local high-velocity tail and the Galactic escape speed. *MNRAS*, 485, 3514. [doi:10.1093/mnras/stz623](https://doi.org/10.1093/mnras/stz623)

Eilers, A.-C., Hogg, D. W., Rix, H.-W., & Ness, M. K. (2019). The Circular Velocity Curve of the Milky Way from 5 to 25 kpc. *ApJ*, 871, 120. [doi:10.3847/1538-4357/aaf648](https://doi.org/10.3847/1538-4357/aaf648)

ESA (1997). *The Hipparcos and Tycho Catalogues*, ESA SP-1200, Vol. 1, §1.5.3.

Freeman, K. C. (1970). On the Disks of Spiral and S0 Galaxies. *ApJ*, 160, 811. [doi:10.1086/150474](https://doi.org/10.1086/150474)

Gaia Collaboration, Prusti, T., et al. (2016). The Gaia mission. *A&A*, 595, A1. [doi:10.1051/0004-6361/201629272](https://doi.org/10.1051/0004-6361/201629272)

Gaia Collaboration, Brown, A. G. A., et al. (2018). Gaia Data Release 2: Summary of the contents and survey properties. *A&A*, 616, A1. [doi:10.1051/0004-6361/201833051](https://doi.org/10.1051/0004-6361/201833051)

Gaia Collaboration, Vallenari, A., et al. (2023). Gaia Data Release 3: Summary of the content and survey properties. *A&A*, 674, A1. [doi:10.1051/0004-6361/202243940](https://doi.org/10.1051/0004-6361/202243940)

GRAVITY Collaboration, Abuter, R., et al. (2022). Mass distribution in the Galactic Center based on interferometric astrometry of multiple stellar orbits. *A&A*, 657, L12. [doi:10.1051/0004-6361/202142465](https://doi.org/10.1051/0004-6361/202142465)

Hernquist, L. (1990). An Analytical Model for Spherical Galaxies and Bulges. *ApJ*, 356, 359. [doi:10.1086/168845](https://doi.org/10.1086/168845)

Kass, R. E., & Raftery, A. E. (1995). Bayes Factors. *JASA*, 90, 773. [doi:10.1080/01621459.1995.10476572](https://doi.org/10.1080/01621459.1995.10476572)

Katz, D., et al. (2023). Gaia Data Release 3: Properties and validation of the radial velocities. *A&A*, 674, A5. [doi:10.1051/0004-6361/202244220](https://doi.org/10.1051/0004-6361/202244220)

Lindegren, L., et al. (2018). Gaia Data Release 2: The astrometric solution. *A&A*, 616, A2. [doi:10.1051/0004-6361/201832727](https://doi.org/10.1051/0004-6361/201832727)

Lindegren, L., et al. (2021). Gaia Early Data Release 3: Parallax bias versus magnitude, colour, and position. *A&A*, 649, A4. [doi:10.1051/0004-6361/202039653](https://doi.org/10.1051/0004-6361/202039653)

McMillan, P. J. (2017). The mass distribution and gravitational potential of the Milky Way. *MNRAS*, 465, 76. [doi:10.1093/mnras/stw2759](https://doi.org/10.1093/mnras/stw2759)

Navarro, J. F., Frenk, C. S., & White, S. D. M. (1996). The Structure of Cold Dark Matter Halos. *ApJ*, 462, 563. [doi:10.1086/177173](https://doi.org/10.1086/177173)

Read, J. I. (2014). The local dark matter density. *J. Phys. G*, 41, 063101. [doi:10.1088/0954-3899/41/6/063101](https://doi.org/10.1088/0954-3899/41/6/063101)

Reid, M. J., & Brunthaler, A. (2020). The Proper Motion of Sagittarius A*. III. *ApJ*, 892, 39. [doi:10.3847/1538-4357/ab76cd](https://doi.org/10.3847/1538-4357/ab76cd)

Schönrich, R., Binney, J., & Dehnen, W. (2010). Local kinematics and the local standard of rest. *MNRAS*, 403, 1829. [doi:10.1111/j.1365-2966.2010.16253.x](https://doi.org/10.1111/j.1365-2966.2010.16253.x)

Ueshima, K. (2010). *Study of pulse shape discrimination and low background techniques for liquid xenon dark matter detectors* (PhD thesis, University of Tokyo). [PDF](https://www-sk.icrr.u-tokyo.ac.jp/xmass/publist/ueshima_PhD.pdf), the source of the original five-component decomposition.

*This work has made use of data from the European Space Agency (ESA) mission Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia Data Processing and Analysis Consortium (DPAC, https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the DPAC has been provided by national institutions, in particular the institutions participating in the Gaia Multilateral Agreement.*
"""

# ╔═╡ ed94aec5-48bf-4f03-86a9-12040996f596
md"""
# Supporting Functions
"""

# ╔═╡ 96ec65b5-f09b-4bf9-9ed4-b65ac7c7d4b2
md"""
### Data access
"""

# ╔═╡ e950cbe1-3cff-458f-8761-d12323f125c4
function adql_dr3_train(n::Integer)
	"""
	SELECT TOP $(n)
	    g.source_id, g.ra, g.dec, g.l, g.b,
	    g.parallax, g.parallax_error, g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
	    g.radial_velocity, g.radial_velocity_error, g.ruwe
	FROM gaiadr3.gaia_source AS g
	WHERE g.radial_velocity IS NOT NULL
	    AND ABS(g.b) < 1
	    AND (g.l < 5 OR g.l > 355 OR (g.l > 175 AND g.l < 185))
	    AND g.parallax_over_error > 5
	    AND g.ruwe < 1.4
	    AND MOD(g.random_index, 2) = 0
	ORDER BY g.random_index"""
end

# ╔═╡ 16885115-5bd5-4e4f-9b9d-94b4d48da6f5
function adql_dr2_test(n::Integer)
	"""
	SELECT TOP $(n)
	    g.source_id, g.ra, g.dec, g.l, g.b,
	    g.parallax, g.parallax_error, g.pmra, g.pmra_error, g.pmdec, g.pmdec_error,
	    g.radial_velocity, g.radial_velocity_error, r.ruwe
	FROM gaiadr2.gaia_source AS g
	JOIN gaiadr2.ruwe AS r ON r.source_id = g.source_id
	WHERE g.radial_velocity IS NOT NULL
	    AND ABS(g.b) < 1
	    AND (g.l < 5 OR g.l > 355 OR (g.l > 175 AND g.l < 185))
	    AND g.parallax_over_error > 5
	    AND r.ruwe < 1.4
	    AND NOT EXISTS (
	        SELECT n.dr3_source_id
	        FROM gaiadr3.dr2_neighbourhood AS n
	        JOIN gaiadr3.gaia_source AS s ON s.source_id = n.dr3_source_id
	        WHERE n.dr2_source_id = g.source_id AND MOD(s.random_index, 2) = 0)
	ORDER BY g.random_index"""
end

# ╔═╡ 86afaa82-8272-4018-b37f-0597921d1c4c
"""
    gaia_tap_query(adql; outfile, timeout = 1800, poll = 5)

Submit `adql` as an asynchronous job to the ESA Gaia Archive TAP service, wait for it
to finish, save the CSV to `outfile` and return it as a `DataFrame`.
"""
function gaia_tap_query(adql::AbstractString; outfile::AbstractString, timeout = 1800, poll = 5)
	base = "https://gea.esac.esa.int/tap-server/tap"
	form = HTTP.URIs.escapeuri(Dict("REQUEST" => "doQuery", "LANG" => "ADQL",
		"FORMAT" => "csv", "PHASE" => "RUN", "QUERY" => adql))
	resp = HTTP.post("$base/async", ["Content-Type" => "application/x-www-form-urlencoded"], form;
		redirect = false, status_exception = false)
	resp.status in (200, 201, 303) || error("TAP submission failed (HTTP $(resp.status)): $(String(resp.body))")
	job = HTTP.header(resp, "Location")
	isempty(job) && error("TAP service did not return a job URL")
	startswith(job, "http") || (job = "https://gea.esac.esa.int" * job)

	t0 = time()
	while true
		phase = strip(String(HTTP.get("$job/phase").body))
		phase == "COMPLETED" && break
		phase in ("ERROR", "ABORTED") && error("Gaia job $phase:\n" * String(HTTP.get("$job/error"; status_exception = false).body))
		time() - t0 > timeout && error("Gaia job timed out after $(timeout) s (last phase: $phase)")
		sleep(poll)
	end

	bytes = HTTP.get("$job/results/result").body
	if length(bytes) > 2 && bytes[1] == 0x1f && bytes[2] == 0x8b     # gzip magic number
		bytes = transcode(GzipDecompressor, bytes)
	end
	mkpath(dirname(outfile))
	write(outfile, bytes)
	return CSV.read(outfile, DataFrame)
end

# ╔═╡ fd83c799-ce7b-4bda-a2c0-e0a6d7208fb8
function load_or_fetch(q, train_path, test_path, force::Bool, log::Vector{String})
	same_query(path, adql) = isfile(path) && isfile(path * ".adql") && read(path * ".adql", String) == adql
	if !force && same_query(train_path, q.train) && same_query(test_path, q.test)
		push!(log, "Loaded cached query results from `$(dirname(train_path))`.")
		return CSV.read(train_path, DataFrame), CSV.read(test_path, DataFrame), :cache
	end
	try
		push!(log, "Queried the Gaia Archive for DR3 (training) and DR2 (test) on $(Libc.strftime("%Y-%m-%d %H:%M", time())).")
		tr = gaia_tap_query(q.train; outfile = train_path)
		write(train_path * ".adql", q.train)
		te = gaia_tap_query(q.test; outfile = test_path)
		write(test_path * ".adql", q.test)
		return tr, te, :archive
	catch err
		push!(log, "Archive query failed: `$(first(sprint(showerror, err), 300))`")
		legacy = joinpath(dirname(train_path), "legacy")
		tr = CSV.read(joinpath(legacy, "DR3Training-1745182203820O-result.csv"), DataFrame)
		te = CSV.read(joinpath(legacy, "DR2Test-1745182248405O-result.csv"), DataFrame)
		tr = tr[iseven.(tr.source_id ÷ 1024), :]   # legacy files lack random_index; bit 10 is well mixed
		return tr, te, :legacy
	end
end

# ╔═╡ 0ce157a0-d3b9-4d42-acd0-1e8279cc1482
"""
    remove_overlap(train, test; radius_arcsec = 1.0)

Drop every test star that lies within `radius_arcsec` of any training star. The 1″
radius comfortably covers the 0.5-yr epoch difference between DR2 (J2015.5) and
DR3 (J2016.0) for all but the very fastest nearby stars.
"""
function remove_overlap(train::DataFrame, test::DataFrame; radius_arcsec = 1.0)
	unitvecs(ra, dec) = permutedims(hcat(cosd.(dec) .* cosd.(ra), cosd.(dec) .* sind.(ra), sind.(dec)))
	tree = KDTree(unitvecs(Float64.(train.ra), Float64.(train.dec)))
	_, chord = nn(tree, unitvecs(Float64.(test.ra), Float64.(test.dec)))
	sep_arcsec = rad2deg.(2 .* asin.(clamp.(chord ./ 2, 0, 1))) .* 3600
	keep = sep_arcsec .> radius_arcsec
	train_ids = Set(train.source_id)
	report = (n_test_in = nrow(test),
		n_same_id = count(in(train_ids), test.source_id),
		n_pos_match = count(!, keep),
		n_test_out = count(keep))
	return test[keep, :], report
end

# ╔═╡ e860e089-f1c5-4af5-8e6c-2d5f63f4dcb6
md"""
### Cuts, kinematics and subsets
"""

# ╔═╡ c5461c90-3b56-4441-bdac-29f181a32fdc
function quality_cuts(df::DataFrame, zp::Real; poe_min = 5.0, ruwe_max = 1.4)
	cols = [:source_id, :ra, :dec, :l, :b, :parallax, :parallax_error, :pmra, :pmra_error,
		:pmdec, :pmdec_error, :radial_velocity, :radial_velocity_error]
	d = dropmissing(df, cols)
	in_window(l) = l < 5 || l > 355 || (175 < l < 185)
	keep = [abs(d.b[i]) < 1 && in_window(d.l[i]) &&
		(d.parallax[i] - zp) / d.parallax_error[i] > poe_min for i in 1:nrow(d)]
	if hasproperty(d, :ruwe)
		keep = keep .& [!ismissing(x) && x < ruwe_max for x in d.ruwe]
	end
	d = d[keep, :]
	d.plx_corr = d.parallax .- zp                       # zero-point correction (mas)
	d.distance_kpc = 1 ./ d.plx_corr
	d.distance_kpc_error = d.parallax_error ./ d.plx_corr .^ 2
	d.proper_motion = hypot.(d.pmra, d.pmdec)
	d.tangential_velocity = 4.740470463533348 .* d.proper_motion .* d.distance_kpc
	d.space_velocity = hypot.(d.tangential_velocity, d.radial_velocity)
	return d
end

# ╔═╡ 0eda1a1e-fea9-4d19-99bb-7d97223a2b43
"Rotate an ICRS Cartesian vector into Galactic coordinates with Gaia's matrix A_G′."
function icrs_to_gal(x, y, z)
	(-0.0548755604162154 * x - 0.8734370902348850 * y - 0.4838350155487132 * z,
	  0.4941094278755837 * x - 0.4448296299600112 * y + 0.7469822444972189 * z,
	 -0.8676661490190047 * x - 0.1980763734312015 * y + 0.4559837761750669 * z)
end

# ╔═╡ fba6977d-7da0-42b3-9529-cbc90a40b328
"""
    galactocentric_kinematics(ra, dec, d_kpc, pmra, pmdec, vr, sun)

Return `(R, z, vphi, vR, vz)` in kpc and km/s for one star. `pmra` is μ_α* (includes cos δ).
"""
function galactocentric_kinematics(ra, dec, d, pmra, pmdec, vr, sun)
	α, δ = deg2rad(ra), deg2rad(dec)
	cα, sα, cδ, sδ = cos(α), sin(α), cos(δ), sin(δ)
	k = 4.740470463533348 * d
	# ICRS velocity = vr r̂ + k (μα* p̂ + μδ q̂)
	vx = vr * cδ * cα + k * (-pmra * sα - pmdec * sδ * cα)
	vy = vr * cδ * sα + k * ( pmra * cα - pmdec * sδ * sα)
	vz = vr * sδ      + k * ( pmdec * cδ)
	x, y, z = icrs_to_gal(d * cδ * cα, d * cδ * sα, d * sδ)
	U, V, W = icrs_to_gal(vx, vy, vz)
	X, Y = x - sun.R0, y
	vX, vY, vZ = U + sun.U, V + sun.Vtot, W + sun.W
	R = hypot(X, Y)
	return (R = R, z = z, vphi = (Y * vX - X * vY) / R, vR = (X * vX + Y * vY) / R, vz = vZ)
end

# ╔═╡ 43b47e81-94cf-4a3c-9336-9c75eb3c3ddc
function add_kinematics!(d::DataFrame, sun; nmc = 32, seed = 416)
	cols = kinematics_loop(Vector{Float64}(d.ra), Vector{Float64}(d.dec), Vector{Float64}(d.plx_corr),
		Vector{Float64}(d.parallax_error), Vector{Float64}(d.pmra), Vector{Float64}(d.pmra_error),
		Vector{Float64}(d.pmdec), Vector{Float64}(d.pmdec_error),
		Vector{Float64}(d.radial_velocity), Vector{Float64}(d.radial_velocity_error), sun, nmc, MersenneTwister(seed))
	d.R, d.vphi, d.vR, d.vz, d.R_error, d.vphi_error = cols
	return d
end

# ╔═╡ 4b407e61-b74e-4d55-9551-f7df1875134f
function kinematics_loop(ra, dec, plx, eplx, pmra, epmra, pmdec, epmdec, vr, evr, sun, nmc, rng)
	n = length(ra)
	R, vφ, vR, vz, σR, σφ = zeros(n), zeros(n), zeros(n), zeros(n), zeros(n), zeros(n)
	sR, sφ = zeros(nmc), zeros(nmc)
	for i in 1:n
		k = galactocentric_kinematics(ra[i], dec[i], 1 / plx[i], pmra[i], pmdec[i], vr[i], sun)
		R[i], vφ[i], vR[i], vz[i] = k.R, k.vphi, k.vR, k.vz
		for j in 1:nmc   # Monte Carlo error propagation
			p = max(plx[i] + eplx[i] * randn(rng), 1e-3)
			kj = galactocentric_kinematics(ra[i], dec[i], 1 / p,
				pmra[i] + epmra[i] * randn(rng), pmdec[i] + epmdec[i] * randn(rng),
				vr[i] + evr[i] * randn(rng), sun)
			sR[j], sφ[j] = kj.R, kj.vphi
		end
		σR[i], σφ[i] = std(sR), std(sφ)
	end
	return R, vφ, vR, vz, σR, σφ
end

# ╔═╡ 9578d0ec-abe7-4796-a4ad-840d792bc625
function finalize_sample(d::DataFrame, sun)
	add_kinematics!(d, sun)
	speed = sqrt.(d.vphi .^ 2 .+ d.vR .^ 2 .+ d.vz .^ 2)
	return d[(d.vphi .> 0) .& (speed .< sun.v_esc) .& (d.R .> 0), :]
end

# ╔═╡ 39372443-e602-42b0-8224-67263e1f7236
function pick_subset(d::DataFrame, k::Integer; seed = 416)
	k == 0 && return d
	idx = shuffle(MersenneTwister(seed), 1:nrow(d))
	return d[sort(idx[k:4:end]), :]
end

# ╔═╡ bc086ab1-1a52-418c-9a87-f43f29dda1b2
function bin_rotation_curve(R, v; width = 0.5, rmax = 20.0, nmin = 20, nboot = 200, rng = MersenneTwister(2025))
	out = DataFrame(R = Float64[], V = Float64[], σ_stat = Float64[], n = Int[])
	edges = collect(0.0:width:rmax)
	for i in 1:length(edges)-1
		m = (R .>= edges[i]) .& (R .< edges[i+1])
		n = count(m)
		n < nmin && continue
		vb = v[m]
		boot = [median(rand(rng, vb, n)) for _ in 1:nboot]
		push!(out, (median(R[m]), median(vb), std(boot), n))
	end
	return out
end

# ╔═╡ 7e74367b-4240-4f87-ac77-843a519571c5
md"""
### Rotation-curve models
"""

# ╔═╡ 44aa0473-696b-4e43-b787-001d108f2167
begin
	const G_GRAV = 4.30091e-6          # kpc (km/s)² / M☉
	v_kepler(R, p) = sqrt.(G_GRAV * p[1] * 1e10 ./ R)             # p = [M / 1e10 M☉]
	v_linear(R, p) = p[1] .+ p[2] .* (R .- 8.277)                 # p = [V₀, dV/dR]; pivot at R₀
	v_simple(R, p) = p[1] .* (1 .- exp.(-p[2] .* R))              # p = [V∞, S]
end

# ╔═╡ ff961d26-45ba-4266-a699-a43420e11196
begin
	v_hernquist(R, M, a) = sqrt(G_GRAV * M * R) / (R + a)

	function v_freeman(R, Σ0, Rd)          # Σ0 in M☉/kpc², Rd in kpc
		R <= 1e-6 && return 0.0
		y = R / (2Rd)
		v2 = 4π * G_GRAV * Σ0 * Rd * y^2 *
			(besseli(0, y) * besselk(0, y) - besseli(1, y) * besselk(1, y))
		return sqrt(max(v2, 0.0))
	end

	function v_nfw(R, ρ0, rs)               # ρ0 in M☉/kpc³
		R <= 1e-6 && return 0.0
		x = R / rs
		return sqrt(4π * G_GRAV * ρ0 * rs^3 * (log1p(x) - x / (1 + x)) / R)
	end

	# Fixed components (McMillan 2017, simplified)
	const BULGE = (M = 9.0e9, a = 0.5)
	const HI_DISK = (Σ0 = 1.1e10 / (2π * 7.0^2), Rd = 7.0)
	const H2_DISK = (Σ0 = 1.2e9 / (2π * 1.5^2), Rd = 1.5)
	const RS_HALO = 19.6

	# Free parameters p = [Σ0 / 1e8 M☉ kpc⁻², R_d / kpc, ρ0 / 1e7 M☉ kpc⁻³]
	function mw_components(R, p)
		(bulge = v_hernquist(R, BULGE.M, BULGE.a),
		 disk  = v_freeman(R, p[1] * 1e8, p[2]),
		 HI    = v_freeman(R, HI_DISK.Σ0, HI_DISK.Rd),
		 H2    = v_freeman(R, H2_DISK.Σ0, H2_DISK.Rd),
		 halo  = v_nfw(R, p[3] * 1e7, RS_HALO))
	end
	vc_total(R, p) = sqrt(sum(abs2, values(mw_components(R, p))))
	v_complex(R, p) = [vc_total(r, p) for r in R]
end

# ╔═╡ 3cc1fcd4-ca4c-49a3-80f3-fc9503cb90ec
chi2(f, p, b::DataFrame, σsys) = sum(abs2.(b.V .- f(b.R, p)) ./ (b.σ_stat .^ 2 .+ σsys^2))

# ╔═╡ 83deb01e-2a48-4cae-a9c9-883695eafc6d
function fit_all_models(b::DataFrame, σsys; Rd_prior = true)
	w = 1 ./ (b.σ_stat .^ 2 .+ σsys^2)
	kepler = curve_fit(v_kepler, b.R, b.V, w, [10.0]; lower = [0.01], upper = [1000.0])
	linear = curve_fit(v_linear, b.R, b.V, w, [230.0, 0.0])
	simple = curve_fit(v_simple, b.R, b.V, w, [230.0, 0.5]; lower = [50.0, 0.01], upper = [500.0, 20.0])

	# Complex model: a sentinel point at R = -1 carries the Gaussian prior on R_d
	history = Vector{Vector{Float64}}()
	function tracked(R, p)
		push!(history, copy(p))
		return [r < 0 ? p[2] : vc_total(r, p) for r in R]
	end
	x = Rd_prior ? vcat(b.R, -1.0) : b.R
	y = Rd_prior ? vcat(b.V, 2.6) : b.V
	wc = Rd_prior ? vcat(w, 1 / 0.5^2) : w
	complex = curve_fit(tracked, x, y, wc, [8.96, 2.5, 0.854];
		lower = [0.1, 0.5, 0.01], upper = [100.0, 10.0, 20.0])
	return (kepler = kepler, linear = linear, simple = simple, complex = complex, history = history)
end

# ╔═╡ 5fbe27c8-a394-492a-a76f-3e81f75d2c4e
fit_table = let b = bins_train, n = nrow(bins_train)
	rows = [("Keplerian", v_kepler, fits.kepler), ("Linear", v_linear, fits.linear),
		("Simple", v_simple, fits.simple), ("Complex", v_complex, fits.complex)]
	DataFrame(map(rows) do (name, f, ft)
		k = length(ft.param)
		c = chi2(f, ft.param, b, σ_sys)
		(model = name, k = k, chi2 = c, dof = n - k, chi2_red = c / (n - k),
		 AIC = c + 2k, BIC = c + k * log(n))
	end)
end

# ╔═╡ 455fea3a-b8f8-45ee-a536-da13d6e92d71
function evaluate_on_test(fits, bt::DataFrame, σsys)
	rows = [("Keplerian", v_kepler, fits.kepler), ("Linear", v_linear, fits.linear),
		("Simple", v_simple, fits.simple), ("Complex", v_complex, fits.complex)]
	DataFrame(map(rows) do (name, f, ft)
		c = chi2(f, ft.param, bt, σsys)
		(model = name, chi2_test = c, chi2_test_per_bin = c / max(nrow(bt), 1))
	end)
end

# ╔═╡ 2a9ad5f5-7549-4cc2-a857-01b1f3efddd5
md"""
### Plotting and reporting helpers
"""

# ╔═╡ 512be5e0-2103-4dc8-a396-dcd964a3476d
function plot_residuals(df::DataFrame, b::DataFrame, f, p; xmax = 20, title = "Residuals: Observed − Model")
	res = df.vphi .- f(df.R, p)
	plt = scatter(df.R, res; ms = 1.5, alpha = 0.12, msw = 0, color = :gray, label = "Individual stars",
		xlabel = "Galactocentric Radius (kpc)", ylabel = "Residual (km/s)", title = title,
		xlims = (0, xmax), ylims = (-120, 120), legendposition = :bottomleft)
	scatter!(plt, b.R, b.V .- f(b.R, p); yerror = b.σ_stat, color = :red, ms = 4, label = "Binned medians")
	hline!(plt, [0]; color = :black, ls = :dash, lw = 2, label = "Zero Residual")
	return plt
end

# ╔═╡ 5391dd2c-6733-463c-8cc6-8f5a7fc31e10
function plot_components(df::DataFrame, p; xmax = 20, title = "Complex Model")
	r = 0.1:0.1:xmax
	comps = [mw_components(ri, p) for ri in r]
	plt = scatter(df.R, df.vphi; ms = 1.5, alpha = 0.1, msw = 0, color = :gray, label = "Stars",
		xlabel = "Galactocentric Radius (kpc)", ylabel = "Orbital Velocity (km/s)", title = title,
		xlims = (0, xmax), ylims = (0, 450), legend = :topright)
	plot!(plt, r, [c.bulge for c in comps]; label = "Bulge", color = :blue, lw = 2)
	plot!(plt, r, [c.disk for c in comps]; label = "Stellar Disk", color = :green, lw = 2)
	plot!(plt, r, [c.HI for c in comps]; label = "HI Layer", color = :gold, lw = 2)
	plot!(plt, r, [c.H2 for c in comps]; label = "H2 Layer", color = :pink, lw = 2)
	plot!(plt, r, [c.halo for c in comps]; label = "Dark Halo", color = :gray40, lw = 2)
	plot!(plt, r, v_complex(r, p); label = "Total", color = :black, lw = 3)
	return plt
end

# ╔═╡ 556c7883-d294-4323-ac75-be55f7788827
function chi2_heatmap(f, b, σsys, yrange, xrange; xlabel, ylabel, title, best)
	M = [chi2(f, [y, x], b, σsys) for y in yrange, x in xrange]
	cmin, idx = findmin(M)
	plt = heatmap(xrange, yrange, log10.(M); xlabel = xlabel, ylabel = ylabel, title = title,
		colorbar_title = "log₁₀ χ²", c = :viridis)
	scatter!(plt, [xrange[idx[2]]], [yrange[idx[1]]]; color = :red, ms = 6, label = "Grid minimum")
	scatter!(plt, [best[2]], [best[1]]; color = :white, marker = :star5, ms = 8, label = "Least-squares fit")
	annotate!(plt, xrange[idx[2]], yrange[idx[1]] + 0.04 * (last(yrange) - first(yrange)),
		text("min χ² = $(round(cmin, digits = 1))", :white, 9))
	return plt
end

# ╔═╡ 6457d7f2-900d-4545-a671-93cf38ae9c42
function model_table_md(t::DataFrame)
	best_bic = t.model[argmin(t.BIC)]
	lines = ["| Model | k | χ² | dof | χ²/dof | AIC | BIC | ΔBIC |", "|:--|--:|--:|--:|--:|--:|--:|--:|"]
	for r in eachrow(t)
		push!(lines, @sprintf("| %s | %d | %.1f | %d | %.2f | %.1f | %.1f | %.1f |",
			r.model, r.k, r.chi2, r.dof, r.chi2_red, r.AIC, r.BIC, r.BIC - minimum(t.BIC)))
	end
	push!(lines, "", "**Preferred model by BIC: $(best_bic).**")
	return join(lines, "\n")
end

# ╔═╡ 86a44141-1779-470d-81dd-efcb7e6cdcf6
function test_table_md(t::DataFrame, nbins::Integer)
	lines = ["### Out-of-sample performance on the independent DR2 stars ($(nbins) bins)",
		"| Model (trained on DR3) | χ² on test | χ² per bin |", "|:--|--:|--:|"]
	for r in eachrow(t)
		push!(lines, @sprintf("| %s | %.1f | %.2f |", r.model, r.chi2_test, r.chi2_test_per_bin))
	end
	nbins < 5 && push!(lines, "", "⚠️ Only $(nbins) test bins survived, too few for a meaningful comparison.")
	return join(lines, "\n")
end

# ╔═╡ b105a025-ce4d-4a4f-80bb-816150abc0ae
function parameter_shift_md(train_fit, test_fit)
	fmt(ft, i, s) = @sprintf("%.2f ± %.2f", ft.param[i] * s, stderror(ft)[i] * s)
	rows = [("Linear V₀ (km/s)", :linear, 1, 1.0), ("Linear slope (km/s/kpc)", :linear, 2, 1.0),
		("Simple V∞ (km/s)", :simple, 1, 1.0), ("Simple S (1/kpc)", :simple, 2, 1.0),
		("Complex Σ₀ (M☉/pc²)", :complex, 1, 100.0), ("Complex R_d (kpc)", :complex, 2, 1.0),
		("Complex ρ₀ (10⁻³ M☉/pc³)", :complex, 3, 10.0)]
	lines = ["### Do the fitted parameters hold up?", "| Parameter | Trained on DR3 | Refit on DR2 test |", "|:--|--:|--:|"]
	for (name, m, i, s) in rows
		a, b = getfield(train_fit, m), getfield(test_fit, m)
		push!(lines, "| $name | $(fmt(a, i, s)) | $(fmt(b, i, s)) |")
	end
	return join(lines, "\n")
end

# ╔═╡ 1d7e1195-3674-446b-80cd-54ea10c682ff
function conclusion_md(t::DataFrame, fits, test_t, source)
	order = sortperm(t.BIC)
	best, second = t.model[order[1]], t.model[order[2]]
	dBIC = t.BIC[order[2]] - t.BIC[order[1]]
	vc0 = vc_total(8.277, fits.complex.param)
	lin = fits.linear.param
	s = """
	Across the training bins, the **$(best)** model is preferred by the Bayesian Information Criterion, ahead of the $(second) model by ΔBIC = $(round(dBIC, digits = 1))$(dBIC > 10 ? ", which counts as very strong evidence" : dBIC > 2 ? ", which is positive but not overwhelming evidence" : ", which is not a decisive difference")). The Keplerian point-mass curve is ruled out by a wide margin, which is the classic signature that the Milky Way's mass keeps growing well beyond its bright center.

	The linear fit gives a circular speed of **$(round(lin[1], digits = 1)) km/s** at the Sun with a slope of **$(round(lin[2], digits = 2)) km/s/kpc**, while the complex model gives \$v_c(R_0)\$ = **$(round(vc0, digits = 1)) km/s**. For context, Eilers et al. (2019) found 229.0 km/s and −1.7 km/s/kpc with a smaller \$R_0\$ and an asymmetric-drift correction, and our adopted \$\\Theta_0\$ is $(round(8.277 * 6.411 * 4.740470463533348 - 12.24, digits = 1)) km/s. Differences of several km/s are expected from the different \$R_0\$, from asymmetric drift, and from our narrow sky windows.
	"""
	if test_t !== nothing
		bt = test_t.model[argmin(test_t.chi2_test)]
		s *= "\n\nOn the independent DR2 stars, the model with the lowest out-of-sample χ² is **$(bt)**. " *
			(bt == best ? "The training and test sets agree on the preferred model, which is exactly what we would hope to see if the result reflects the Galaxy rather than one sample." :
			"The training and test sets **disagree** on the preferred model. This is an honest null result: the test set is smaller and noisier, and it tells us that the data cannot yet cleanly separate these models on their own.")
	else
		s *= "\n\n*Tick the test checkbox above to add the out-of-sample comparison to this conclusion.*"
	end
	source == :legacy && (s *= "\n\n⚠️ These conclusions were computed from the offline legacy files and should be re-run with a live archive query.")
	return s
end

# ╔═╡ 2258a570-ee09-44c3-9d69-303f2478c0de
function mass_plots(p)
	r = range(1.0, 20.0; length = 300)
	x(R) = R / RS_HALO
	M_halo(R) = 4π * p[3] * 1e7 * RS_HALO^3 * (log1p(x(R)) - x(R) / (1 + x(R)))
	M_bulge(R) = BULGE.M * R^2 / (R + BULGE.a)^2
	M_disks(R) = let c = mw_components(R, p); R * (c.disk^2 + c.HI^2 + c.H2^2) / G_GRAV end
	M_tot(R) = R * vc_total(R, p)^2 / G_GRAV
	f_dm(R) = mw_components(R, p).halo^2 / vc_total(R, p)^2

	p1 = plot(r, [M_bulge.(r) .+ M_disks.(r), M_halo.(r), M_tot.(r)];
		label = ["Baryonic (bulge + disks)" "Dark matter (NFW)" "Total (R v²/G)"],
		xlabel = "Radius (kpc)", ylabel = "Enclosed Mass (M☉)", title = "Enclosed Mass Profiles",
		yscale = :log10, lw = 2, legend = :bottomright)
	vline!(p1, [8.277]; ls = :dot, color = :black, label = "Sun")
	p2 = plot(r, f_dm.(r); label = "f_DM = v²_halo / v²_c", xlabel = "Radius (kpc)",
		ylabel = "Dark matter fraction", title = "Dark Matter Fraction vs Radius", lw = 2,
		ylims = (0, 1), color = :purple)
	vline!(p2, [8.277]; ls = :dot, color = :black, label = "Sun")
	return plot(p1, p2; layout = (1, 2), size = (900, 380))
end

# ╔═╡ cdc5e523-6cde-4ea7-81ae-121512095d2d
function mass_summary_md(ft, sun; ndraw = 500, rng = MersenneTwister(7))
	p = ft.param
	C = Symmetric(estimate_covar(ft))
	L = cholesky(C + 1e-12I).L
	draws = [max.(p .+ L * randn(rng, 3), 1e-6) for _ in 1:ndraw]
	x0 = sun.R0 / RS_HALO
	ρdm(q) = q[3] * 1e7 / (x0 * (1 + x0)^2) * 3.797e-8        # M☉/kpc³ → GeV/cm³
	Md(q) = 2π * q[1] * 1e8 * q[2]^2
	fdm(q) = mw_components(sun.R0, q).halo^2 / vc_total(sun.R0, q)^2
	q(f) = (median(f.(draws)), std(f.(draws)))
	a, b, c = q(ρdm), q(Md), q(fdm)
	"""
	| Derived quantity | Value | Literature |
	|:--|--:|:--|
	| Local dark matter density ρ_DM(R₀) | $(@sprintf("%.2f ± %.2f", a[1], a[2])) GeV/cm³ | 0.2–0.6 GeV/cm³ (Read 2014; de Salas & Widmark 2021) |
	| Stellar disk mass M_d | $(@sprintf("%.1e ± %.1e", b[1], b[2])) M☉ | ≈ 4–6 × 10¹⁰ M☉ (Bland-Hawthorn & Gerhard 2016) |
	| Dark matter fraction at the Sun f_DM(R₀) | $(@sprintf("%.2f ± %.2f", c[1], c[2])) | |

	Uncertainties are statistical only, from 500 draws of the fit covariance, and do not include the fixed bulge, gas or halo scale radius. A disk mass above the literature range paired with a low local dark matter density is the classic signature of the disk–halo degeneracy; ticking the \$R_d\$ prior above tightens this, but only data at larger radii can truly break it.
	"""
end

# ╔═╡ Cell order:
# ╟─8005bf0e-01ff-42da-967e-414840eb3046
# ╠═f535728b-a9ce-437a-9a0a-dd38945d9040
# ╟─823860b3-4b10-4180-96a0-3539bbaa451f
# ╠═9f3c700a-a493-4279-a390-9963715dad5d
# ╟─8fb686c1-09fc-41d9-97ee-cf54c4e61692
# ╟─ed5e2cfa-a0be-4779-9500-a6a5fc875c1d
# ╠═a941e758-0c6d-46af-8712-ea3dcbcc32b0
# ╟─724a3a8a-8300-4fcb-bda5-957837ef06df
# ╠═9a8fcaf0-6384-490b-bfe0-305d3943a2ec
# ╟─cf5bd8c7-d1a2-4b25-8211-01a8e9165142
# ╠═93073b09-122a-48e2-bfbf-56234bde9768
# ╟─5ad4cacc-cce9-4c30-aa05-85d8c07a7096
# ╠═7cc76540-5367-43de-815a-23fa9cf9a965
# ╟─b1bd2968-b25a-49a1-a92f-2b48e3fc39f9
# ╠═b9fbde21-9a8b-485e-9c3d-35197d7b753d
# ╟─e2675a61-2a64-49df-99e2-b05f38207c84
# ╠═31c7f62d-ca7e-4cef-9a27-0d842d79fffd
# ╟─a4e08e3f-ea98-4789-bb71-88b68e3e6ace
# ╠═da3aa2fd-9caf-4e0f-ab1e-ac8430728af1
# ╟─a4399939-2c92-4e72-a969-5ae1d99adb44
# ╠═d36d20a6-1924-4db9-9271-42bc0c74a748
# ╟─d14f8cd4-fa6c-4957-9cf4-6f0301ca62b1
# ╟─c84013f0-b8e2-4677-8f42-cf3906aa761d
# ╟─ff92d8a6-caa4-4b3f-ae58-dec7062dab73
# ╟─8da928c0-6daa-4fc8-bdcd-19905055d9e6
# ╟─d3148eaf-58c5-4669-ac47-bc8d855db6f2
# ╠═eda647a4-53ed-4b5a-ae7e-296c2de6f109
# ╟─23083e31-93a1-4c72-ab8c-1de6644b82b2
# ╟─4bf49b31-1ad1-4dfd-9348-c441860aec42
# ╠═5a3b51e1-cf54-42a3-9e77-def601bd0916
# ╠═96a4de86-2506-4076-be57-20afee55989a
# ╠═7bdaabf2-8f4b-4937-a6f0-1401bdccebbb
# ╟─b36d7545-7ed8-4b03-b8b3-95b8a9e1fdd2
# ╟─f5a7f3a0-4585-4873-8bbb-2d1d8ad5443d
# ╟─ada28aba-9c83-4eb7-8a17-d156ca834281
# ╠═aabbc3b5-bb04-4d43-b65f-4841d0d45770
# ╠═efabe12c-ba23-4fc5-9963-084ca7e5f397
# ╠═4ff6dae4-dcc1-40ec-b370-269ed0fc818d
# ╟─42c798a2-e34c-48d5-8321-70f5304fe008
# ╟─03dfef59-8ee7-4f66-b691-4d5d1ebeea1c
# ╟─31d76851-80e8-4ace-a243-bf17bcb4ee6b
# ╠═97a0da3b-4b17-494e-9753-a12cf6f8787e
# ╠═e627ac13-0bf8-4543-99e2-5d5f4abf83ca
# ╠═90e663bc-45aa-4395-96c7-08013b0f44fa
# ╠═5449a8c2-3f18-44c5-a8c1-f4b4ac51fd33
# ╠═ff1628b7-9d49-4095-9b1d-63ad0665c723
# ╠═d5f074d7-a034-4af7-9404-70afc5ae86b5
# ╟─b0fefd6e-5d3f-42de-8e89-2068668a94e3
# ╠═bc07b054-332e-4f48-bffb-f7da596ca249
# ╟─e0901014-ef95-4853-83ce-78fba6be3726
# ╟─c9ab4f74-5ce4-44d2-a840-4f1e971efca6
# ╠═1e9a355e-ccf2-4a36-a95c-907a7806e729
# ╠═177d7ff2-d445-4b53-91a1-f16fd28414de
# ╠═7350f379-fc0a-4440-a729-8fb59e0a9b6d
# ╠═1a723d6c-64d0-41ef-9714-102f6713416b
# ╠═b4ed7185-f35a-4cc9-bdbf-9d18d3c3ef00
# ╟─abaa2538-21b2-4744-aff4-3e8fb18e31a2
# ╠═d1eba40e-a444-4017-b884-9d3d2eca4b0f
# ╠═01c86301-14f5-4926-953d-c05f00759479
# ╠═4ad4950d-2089-4109-b710-44ac69e8fa1a
# ╟─420cf99f-649d-4a75-990c-dce52d9d52a3
# ╟─d8a71c50-d19e-46d9-96ce-57705039cdf0
# ╟─e02ed27e-58c2-42b7-b4d0-fe7ce368ba38
# ╠═a5e796de-2448-4de6-a577-361d84801f1b
# ╠═63018b7a-b3a5-41a3-b29c-9e6030597191
# ╠═31e58353-aa5d-4623-a02f-344eb26ef5d0
# ╠═c7150c60-a74d-4e0f-9fbe-84684e69fe8e
# ╠═a9e6ba8f-8f72-4efc-8adf-aca92a666400
# ╠═93f0caf3-b524-4fa8-b028-cdf96623dcd4
# ╠═a0dce3bf-8ef7-464f-b9c6-8b519732a812
# ╟─69cc2f30-ad9a-42d5-8439-fe710a2e8024
# ╟─40ecc835-4ebd-45c4-94bb-8b3951137baa
# ╠═40a57abc-8b62-461a-890d-4e60a207eaed
# ╠═f1dc2f6e-5d16-4d0e-ae52-2556860fc693
# ╟─94a06dad-e077-4ca5-a249-e151b9acab1c
# ╠═29038359-ce88-45ff-82b3-1863379e1880
# ╠═4722e1b0-e060-4aec-9525-357e2b0ca5fb
# ╠═149f0079-875e-4ce0-a48b-1915d40b0aef
# ╠═37806728-128f-40cb-a339-92d4efcea87b
# ╠═a19ee871-c665-4daf-920f-86882a93610e
# ╟─77fd543a-1eb7-4458-b83f-8f631ce3c927
# ╟─c51ca757-c00c-4603-b1c4-cdccd0833d2b
# ╠═75b540af-e42d-4c72-a616-d182d3a5bfff
# ╟─20d66522-b432-40a8-a64f-d5e8caac02ce
# ╟─673bfffb-e887-4a1a-b7db-40405c76e6d5
# ╠═ce3c8382-9ec2-42a3-87c4-52d83ba0880c
# ╠═7779c5c9-5637-4a0e-9d5e-44fa8b67dff4
# ╠═101356f0-000b-4f14-9fba-881ef7f0fd0f
# ╟─09b90ac7-3525-444e-8775-477a47b24b5c
# ╟─7fa4dbd3-4898-4dc2-a5a3-5f85569d0061
# ╟─ed94aec5-48bf-4f03-86a9-12040996f596
# ╟─96ec65b5-f09b-4bf9-9ed4-b65ac7c7d4b2
# ╟─e950cbe1-3cff-458f-8761-d12323f125c4
# ╟─16885115-5bd5-4e4f-9b9d-94b4d48da6f5
# ╟─86afaa82-8272-4018-b37f-0597921d1c4c
# ╟─fd83c799-ce7b-4bda-a2c0-e0a6d7208fb8
# ╟─0ce157a0-d3b9-4d42-acd0-1e8279cc1482
# ╟─e860e089-f1c5-4af5-8e6c-2d5f63f4dcb6
# ╟─c5461c90-3b56-4441-bdac-29f181a32fdc
# ╟─0eda1a1e-fea9-4d19-99bb-7d97223a2b43
# ╟─fba6977d-7da0-42b3-9529-cbc90a40b328
# ╟─43b47e81-94cf-4a3c-9336-9c75eb3c3ddc
# ╟─4b407e61-b74e-4d55-9551-f7df1875134f
# ╟─9578d0ec-abe7-4796-a4ad-840d792bc625
# ╟─39372443-e602-42b0-8224-67263e1f7236
# ╟─bc086ab1-1a52-418c-9a87-f43f29dda1b2
# ╟─7e74367b-4240-4f87-ac77-843a519571c5
# ╟─44aa0473-696b-4e43-b787-001d108f2167
# ╟─ff961d26-45ba-4266-a699-a43420e11196
# ╟─3cc1fcd4-ca4c-49a3-80f3-fc9503cb90ec
# ╟─83deb01e-2a48-4cae-a9c9-883695eafc6d
# ╟─5fbe27c8-a394-492a-a76f-3e81f75d2c4e
# ╟─455fea3a-b8f8-45ee-a536-da13d6e92d71
# ╟─2a9ad5f5-7549-4cc2-a857-01b1f3efddd5
# ╟─512be5e0-2103-4dc8-a396-dcd964a3476d
# ╟─5391dd2c-6733-463c-8cc6-8f5a7fc31e10
# ╟─556c7883-d294-4323-ac75-be55f7788827
# ╟─6457d7f2-900d-4545-a671-93cf38ae9c42
# ╟─86a44141-1779-470d-81dd-efcb7e6cdcf6
# ╟─b105a025-ce4d-4a4f-80bb-816150abc0ae
# ╟─1d7e1195-3674-446b-80cd-54ea10c682ff
# ╟─2258a570-ee09-44c3-9d69-303f2478c0de
# ╟─cdc5e523-6cde-4ea7-81ae-121512095d2d
