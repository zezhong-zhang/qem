Analysis
--------

In this panel, several procedures are available for using the fitted
model parameters for further analysis, such as atom counting, column
coordinate analysis, and creating (simple) three-dimensional (3D)
structural models.

General Options
~~~~~~~~~~~~~~~


Select new model from MAP
^^^^^^^^^^^^^^^^^^^^^^^^^

When the model-selection method by maximum a posteriori (MAP)
probability has been applied previously, one has the option to choose a
different model and corresponding number of atomic columns for further
analysis by pressing *Select new model from MAP*. A new model can be
selected by using the cursor for indicating the desired number of atomic
columns from the relative probability curve. More details can be found
in section `4.2.3 <#sec:runMAP>`__.

Show models from MAP
^^^^^^^^^^^^^^^^^^^^

This option allows to open an extra screen (*Overview MAP*), showing an
overview of the optimised models and corresponding number of atomic
columns. This screen also opens automatically when pressing the *Select
new model from MAP* button, as discussed in the previous section
`5.1.1 <#sec:newmod>`__. This overview screen serves as a helping tool
for deciding what model should be chosen for further analysis.

Select columns in image
^^^^^^^^^^^^^^^^^^^^^^^

By hitting the *Select columns in image* button, one can indicate a
region of the image to take into account for further analysis. Columns
located outside this region are neglected. The user defines the corner
points of the selected area one by one. Right clicking connects the last
defined point with the starting point. By pressing the *escape* key, the
routine is aborted.

Select columns in histogram
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This option allows to exclude outliers in the histogram of scattering
cross-sections. First, the lower limit must be defined, then the upper
limit. The scattering cross-sections of the columns that fall outside
the limit are neglected for further analysis. By pressing the *escape*
key, the routine can be aborted.

Select columns on type
^^^^^^^^^^^^^^^^^^^^^^

By pressing *Select columns on type*, columns can be selected for
further analysis based on their column type labels, given in the
*Preparation* panel (see section `3 <#chap:prep>`__).


Index columns
~~~~~~~~~~~~~

Here, one can index the columns in terms of distance from a reference
point. This operation should be done before performing a column
coordinate based analysis (see section `5.7 <#strain>`__) or creating a
3D model (see sections `5.3 <#atomCountStat:3Dmodel>`__ and
`5.4 <#atomCountLib:3Dmodel>`__).


Projected unit cell
^^^^^^^^^^^^^^^^^^^

.. figure:: /imgs/ProjUnitZ.png 

    A projected unit cell can be made by defining the projected
    lattice constants (a and b) and the relative positions of the columns.
    Additional information can be given on the depth location (z) of the
    atoms in the columns.

A necessary input to index columns is the location of the different
column positions in a projected unit cell, mentioned in section
`3.3.1 <#input:PUC>`__. If one intends to make a 3D model later on,
z-information must be given, as shown in Figure `5.1 <#fig:projUC>`__.
Then, it becomes possible to define the lattice parameter in c-direction
and the depth locations of the atoms in each column. Atoms in a column
can be added and removed by the buttons *New* and *Delete*,
respectively. All atoms, except the first one, can be deleted by using
the *Clear* button. Each column can be selected from the drop-down menu
in the top-right corner. Make sure that in StatSTEM you provide the
correct pixel size of the image (see section `3.4.1 <#sec:pix>`__) as
the lattice parameters in the projected unit cell should be close to the
experimental values. It is noted that a database is available in
StatSTEM for some common materials and viewing directions.

Start indexing
^^^^^^^^^^^^^^

With *Start indexing*, all coordinates are indexed as a function of
distance in unit cells from a reference coordinate. For this procedure,
a reference coordinate must first be chosen. Then, lattice directions
are searched and the indexing procedure starts. If the automatic
routines in StatSTEM fail, you can guide StatSTEM by using the advanced
options listed below. The details of the automatic routine are described
in Appendix `[strain:procedure] <#strain:procedure>`__.

Reference coordinate
^^^^^^^^^^^^^^^^^^^^

With this option, a reference coordinate can be chosen and a
displacement map is made from this coordinate. Furthermore, the
reference coordinate gets the index (0,0) during the creation of the
strain map. One can choose between different column types for selecting
a reference coordinate. By default, StatSTEM uses the *Most central*
coordinate as a reference, this can be changed by manually selecting
another coordinate by indicating *User defined*.

Direction a lattice
^^^^^^^^^^^^^^^^^^^

For finding the direction of the lattice, an *Automatic* routine can be
used or a manual (*User defined*) input can be given. The automatic
routine uses the projected unit cell parameters to identify the lattice
direction. Here, the distance of the neighbouring coordinates with
respect to the reference coordinate is compared to the given lattice
parameters in the projected unit cell. Once the direction is found, the
lattice parameters are automatically improved by fitting. This option
can be disabled by ticking off *Improve values by fitting*. When this
setting is enabled, a box of N unit cells (standard 3 unit cells (UCs)
to each side) around the reference coordinate is used for finding the
values of the a (and b) lattice parameter in the image. This option is
advised to be used, as the pixel size recorded by an electron microscope
is not always very accurate. Be, however, aware that this option changes
the values of a and b and should only be turned off when you are 100 %
sure about the pixel size.

Atom counting - Statistical
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: /imgs/StatSTEMwithICL.png 

    The ICL criterion evaluated for an experimental image. A local
    minimum appears at 10 Gaussian components.

In HAADF STEM imaging, a statistics-based method has been developed to
count the number of atoms based on the scattering cross-sections (the
total intensities of electrons scattered by the atomic columns), which
increase monotonically with thickness [@Martinez2015]. **Be aware that
this method is only reliable when only one column type is present!** In
the statistics-based method, the scattering cross-sections are presented
in a histogram. Owing to a combination of experimental noise and
residual instabilities, broadened - rather than discrete - components
are observed in such a histogram. Therefore, these results cannot
directly be interpreted in terms of number of atoms. By evaluation of
the so-called integration classification likelihood (ICL) criterion in
combination with Gaussian-mixture model estimation, the number of
components and their locations can be found. From the estimated
locations of the components, the number of atoms in the columns can be
quantified. More information on this methodology can be found in Refs.
[@VanAert2011; @VanAert2013; @DeBacker2013].

Pre-analysis
^^^^^^^^^^^^

In order to evaluate the ICL criterion, an upper limit on the number of
components must be given, which can be specified by *Max components*. Up
to the provided number of components, Gaussian mixture models are fitted
to the histogram of scattering cross-sections and the ICL criterion is
determined. A rough estimate for this upper limit can be obtained by
using the shape of the particle under study.

.. figure:: /imgs/StatSTEMwithAtomCounts.png 

    The atom counts of the atomic columns determined from an
    experimental HAADF STEM image of a Pt/Ir particle.

Post-analysis
^^^^^^^^^^^^^

In StatSTEM, the statistical atom-counting analysis can be peformed by
hitting *Run ICL*. When the Gaussian-mixture model has been fitted to
the histogram of the scattering cross-sections and the ICL calculation
has been performed for each number of components, a suitable number of
components needs to be selected by using the cursor. In the
statistics-based atom-counting procedure, one searches for local minima
in the ICL curve (see Figure `5.2 <#fig:ICL>`__). After a local minimum
is selected, the experimental image with the corresponding atom counts
is shown (see Figure `5.3 <#fig:ATcounts>`__). The calculation can be
aborted by pressing *Stop function*. Once the procedure is aborted,
StatSTEM shows a message asking whether the user wants to select a local
minimum in the current ICL graph. Counting results can be rescaled by
providing a *Counting offset*. This is particularly useful for thick
particles in which no thin columns are present. Another local minimum
from the ICL curve can be selected by pressing the *Select new ICL
minimum* button.

Create 3D model
^^^^^^^^^^^^^^^

.. figure:: /imgs/StatSTEMwith3Dmodel.png

    A 3D model of a Pt/Ir particle.

When the atoms are counted per atomic column and the columns are indexed
(see section `5.2 <#indexCol>`__), a 3D model can be made. Here, the
atoms are distributed symmetrically along the z-direction. **Note that
this 3D model is only a simple model for visualising the 3D shape of the
particle. It should not be used as a final result!** Note that for this
procedure the projected unit cell should contain z-information (see
section `5.2.1 <#strain:PUC>`__). The 3D model that is shown is colour
coded, meaning that each atom type is displayed in a different colour.

Atom counting - Simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Another way of performing atom counting is by directly comparing
libraries of simulated scattering cross-sections with experimental
cross-sections.

Match with simulations
^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /imgs/StatSTEMwithSCSvsThick.png 

    Comparison between the scattering cross-sections obtained by
    the statistics-based method (Experiment) and image simulations (Library).

By clicking on this button, a library containing simulated scattering
cross-sections is asked to be loaded. The atom counts in the image under
investigation are computed by comparing the measured cross-sections from
the fitted model to the loaded library of simulated cross-sectional
values. A MATLAB (*.mat*) or text (*.txt*) file can be loaded containing
the simulated values of the scattering cross-sections. The
cross-sections must be provided in function of column thickness,
expressed as a column vector. The atom counts following from the
simulation-based method can be visualised by indicating *Lib Counts* in
the *Image Options* panel at the right-hand side of the GUI, as opposed
to selecting *Atom Counts* for the statistics-based method. By selecting
*SCS vs. Thickness* from the *Select Image* panel, the scattering
cross-sections obtained by the statistics-based method are compared to
the simulated cross-sectional values (see Figure
`5.5 <#fig:library>`__).


Create 3D model
^^^^^^^^^^^^^^^

This button has the exact same functionality as explained before in
section `5.3.3 <#sec:3dmod>`__ and is used for quickly building a 3D
visualisation of the material structure based on the atom counts and
indexed columns.


Atom counting - Time series
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: /imgs/StatSTEMwithObservationsHMM.png 

    An example of a time series of ADF STEM images of a Pt particle.

When the goal is to perform atom counting from a time series of ADF STEM
images, this functionality can be used. The methodology makes use of a
hidden Markov model (HMM) to determine the number of atoms in each time
frame by explicitly taking into account the possibility of structural
changes. More information on this methodology can be found in Refs.
[@Dewael2020] and [@Dewael2021]. At the moment, preprocessing still
happens outside StatSTEM, but in a later release this will be fully
included. The required format is an inputStatSTEM_HMM structure
containing all images, all coordinates, and all models of the images in
the time series. Using a slider, different images of the time series can
be displayed. Most importantly, the scattering cross-sections determined
from each image are added as a row in a matrix O, which contains the
observations. It is important to sort all scattering cross-sections in
the same order throughout the time series, and in the same order as the
input coordinates.

Run HMM
^^^^^^^

By clicking this button, the hidden Markov model is estimated from the
matrix O containing the observed sequence of scattering cross-sections
for all columns in the ADF STEM images of the time series. The output is
a state sequence, obtained by the Viterbi algorithm, and therefore
called H_viterbi in the output structure, which can be saved from
StatSTEM. This is a matrix with the same dimensions as the observed
sequence, which contains the corresponding number of atoms in the
columns. Atom counts are displayed in StatSTEM, and can be viewed for
different images using a slider.

.. figure:: /imgs/StatSTEMwithAtomCountsHMM.png 

    Atom counts for a time series of ADF STEM images of a Pt particle obtained by using the hidden Markov model.

3D model
~~~~~~~~

.. figure:: /imgs/CoorNum.png 

    The coordination number of (a) an FCC and (b) a BCC crystal can be calculated by searching for neighbouring atoms within a radius of :math:`a/\sqrt{2}` or :math:`2\times a/\sqrt{3}`, respectively.

.. figure:: /imgs/StatSTEMwith3DmodelCoorNum.png 

    A 3D model of a Pt/Ir particle, indicating the coordination number per atom.

In sections `5.3 <#atomCountStat:3Dmodel>`__ and
`5.4 <#atomCountLib:3Dmodel>`__, it has been mentioned that a 3D model
can be made from atom-counting results. Once the model is constructed,
this panel provides options to export the coordinates as an XYZ file or
to calculate the coordination number.

Save model as XYZ
^^^^^^^^^^^^^^^^^

=================== ========= =================== =========
Coordination number Atom type Coordination number Atom type
=================== ========= =================== =========
1                   V         7                   Lu
2                   Mg        8                   Yb
3                   Au        9                   Al
4                   Na        10                  Np
5                   Se        11                  Ho
6                   Zr        12                  Co
=================== ========= =================== =========

Atom type per coordination number that is used when storing the 3D model
with coordination numbers as an XYZ file.

.. figure:: /imgs/StatSTEMwithLattice.png 

    The lattice parameters a and b of a Pt/Ir nanoparticle. Plotting the a and b lattice parameters as a function of distance in the a- or b-direction is possible, as indicated by the red circle.

With this option, the constructed 3D model can be saved as an XYZ file
that can be loaded into other software packages such as Vesta or Visual
Molecular Dynamics (VMD).

Coordination number
^^^^^^^^^^^^^^^^^^^

This function can be used to determine the coordination number of each
atom in the 3D model. The coordination number is determined by
calculating the number of neighbours of each atom within a specific
radius.

Radius
^^^^^^

.. figure:: /imgs/StatSTEMwithCentralShift.png 

    Displacement map of the central atom in :math:`PbCsBr_3` In StatSTEM, the standard radius that is used is :math:`a\times 0.8`, which is a little bit larger than :math:`a/\sqrt{2}` to compensate for small fluctuations in the atom positions.

Number of atoms
^^^^^^^^^^^^^^^

By default, the coordination number is determined for all atoms (100 %).
As this is a demanding calculation, the user can decide to leave out the
most central atoms for computational purposes. The coordination numbers
of these atoms are determined based on the distance from the centre of
the particle. In this manner, one can calculate the coordination number
only for a fraction of the atoms in the particle.

Save coor number as XYZ
^^^^^^^^^^^^^^^^^^^^^^^

Once the coordination numbers are determined, they can be saved as an
XYZ file by hitting *Save coor number as XYZ*. Hereby, a specific atom
type is given in function of the coordination number. The types as a
function of coordination number are listed in Table
`5.1 <#tab:coorNum>`__.


Strain and more
~~~~~~~~~~~~~~~

.. figure:: /imgs/StatSTEMwithOctaTilt.png 

    Octahedral tilt measured along the b-direction in :math:`PtTiO_3`.

In this panel, one can use the fitted atom column coordinates of the
model for determining the lattice parameters, measuring displacement of
atoms, analysing octahedral tilt, and constructing strain maps.

Lattice of type
^^^^^^^^^^^^^^^

This function determines the lattice parameter per column type.
Automatically, a coloured plot is made that shows the lattice
parameters, as illustrated in Figure `5.8 <#strain:latt>`__. StatSTEM
also enables in the *Select Image* panel at the right-hand side of the
GUI to visualise plots where the lattice parameters as a function of
distance from the reference coordinate are shown, in both a- and
b-directions.

Show shift central atom
^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /imgs/StatSTEMwithDisp.png 

    Displacement map of a :math:`PbCsBr_3` particle.

By clicking on this button, a displacement map is made of the central
atom in a unit cell, based on the column indexing described in section
`5.2 <#indexCol>`__. By using the values of the projected unit cell, the
expected coordinates are calculated. The displacement map is generated
by comparing the expected coordinates with the measured coordinates.

Calculate octahedral tilt
^^^^^^^^^^^^^^^^^^^^^^^^^

In perovskite materials, there are oxygen octahedra present surrounding
the B-cation (or A-cation). Due to internal strain, the oxygen octahedra
can rotate. From the indexed columns, octahedral tilt can be determined
when there are atoms present at the relative positions in the unit cell:
(0,0.5) and (0.5,0). If this condition is satisfied, the *Calculate
octahedral tilt* button becomes available to determine the octahedral
tilt as a function of distance in the a- and b-direction. The distance
is measured from the reference coordinate selected when indexing the
columns, as described in section `5.2 <#indexCol>`__. In the plot, the
octahedral tilt is calculated in regions where it is assumed that the
octahedral tilt is alternating between a clockwise and an anti-clockwise
rotation.

Make displacement map
^^^^^^^^^^^^^^^^^^^^^

.. figure:: /imgs/StatSTEMwithStrain.png 
    
    A :math:`\epsilon_{xx}` strain map of a Pt/Ir particle.

By hitting this button, a displacement map is constructed based on the
column indexing described in section `5.2 <#indexCol>`__. By using the
values of the projected unit cell, the expected coordinates are
calculated. The displacement map is generated by comparing the expected
coordinates with the measured coordinates.

Make strain map
^^^^^^^^^^^^^^^

By clicking *Make strain map*, the :math:`\epsilon_{xx}`, :math:`\epsilon_{xy}`,
:math:`\epsilon_{y}` and :math:`\omega_{xy}` strain maps are generated. Hereby, the derivative of the displacement map is used [@Galindo2007]. An example is shown in Fig.
`5.12 <#fig:strain>`__.




