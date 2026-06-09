# JAXiverse

```{admonition} Planned package; API not stable
:class: warning
JAXiverse is not part of the first public StringForge release.  This page
records the intended ecosystem boundary only.  StringForge does not install,
import, or guarantee any JAXiverse API in this release.
```

**Planned role:** multi-axion effective field theory from string compactification
data.

JAXiverse is intended to consume stabilised geometric/EFT data and compute axion
spectra, decay constants, mixing matrices, and couplings.  StringForge will
provide shared data conventions and persistence, not the axion solver itself.

## Planned ownership

- Multi-axion EFT objects and numerical strategies.
- Axion spectra, decay constants, kinetic mixing, and couplings.
- Bridges from future Kähler-sector data to axion observables.

## Current release status

- Not a StringForge dependency.
- Not imported by `stringforge`.
- No install command or executable tutorial cells are provided until public
  release.
- Public documentation will be linked once the package is released.

## Related packages

- [`cytools`](cytools) for geometric input.
- [`kahlerjax`](kahlerjax) as a planned upstream Kähler-sector package.
- [`jaxpolylog`](jaxpolylog) transitively through physics-sector packages.
