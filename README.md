Normalization for probabilistic inference with neurons
======================================================

See the publication at
http://www.springerlink.com/content/j7117u2675r27jv0/
for details.

Like other methods, the important step of normalization of the
probability density functions that are represented in our neural
implementation is left to other mechanisms. Recently we have been
working on methods to include normalization in the inference
transformation (which takes place in the connection weights). We have
recently submitted this work for publication. The supporting Nengo and
Matlab code is in this repository.

Open the `Normalization.3k.good.nef` with Nengo 1.4, and run it to
regenerate the data in the paper. To put an input into the model, run
the `normalization_input_reader.py` script, which takes the
`input_matrix.txt` and generates a node that will give the correct
input.

To generate the plots, run `nengo_plotgen.m` (read instructions in
that file).

Just to get the 'ideal' solution (no neurons), run the
`normalization_noneurons_paper.m`, which is the easiest.
