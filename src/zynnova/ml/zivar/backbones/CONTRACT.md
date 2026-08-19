# ZIVAR local-backbone contract v1

A backbone owns only local representation and short-range scalar energy. Its
adapter must return:

- `energy`: graph scalar with shape `[B]`;
- `node_energy`: optional atomic partition with shape `[N]`;
- `invariant_features`: O(3)-invariant features with shape `[N,F]`.

Input graphs must be complete reciprocal directed radius graphs. For each
`i -> j` image edge with Cartesian shift `S`, the graph contains `j -> i`
with shift `-S`. Pair-energy modules half-count directed entries under this
explicit contract.

Required capabilities are invariant energy/features, conservative forces,
differentiable stress, periodic boundaries, second-order force training and
global callback deployment. `maximum_ell` may be 0...4: the shared framework
supports ell=4, while scalar convolution deliberately advertises ell=0.

The adapter must expose fixed atomic-number ordering and cutoff. Kind,
architecture, implementation, feature dimension, capabilities and element
table are fingerprinted in a checkpoint manifest. Replacing the backbone after
construction or loading a mismatched manifest is an error.

Third-party extensions register an explicit builder through the
`zynnova.zivar_backbones` entry-point group. A local adapter must not implement
charge allocation, magnetism, oxidation states, multipoles, long-range
electrostatics or materials workflows; these remain in the shared ZIVAR layer.

Native MLIAP export is an optional capability and contains only local energy
and force. The complete global model always remains available through the
LAMMPS callback bundle.
