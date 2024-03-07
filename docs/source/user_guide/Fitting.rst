Model Fitting
-------------

The *Fit Model* panel allows to model the loaded image by a
superposition of Gaussian peaks, describing the projected atomic
columns. StatSTEM offers to do this in two ways. There is the so-called
standard procedure, where the model-fitting routine is performed based
on the starting atom column coordinates provided in the *Preparation*
panel. Another, more general, approach is also available. This method
starts from the provided initial atom column coordinates and
automatically searches for more possible column locations. As such, the
model-fitting procedure can be performed without a priori specifying
where the atomic columns are expected to be located. This method can be
especially useful for quantifying images exhibiting low signal-to-noise
ratio (SNR) and low contrast.


Standard Procedure
~~~~~~~~~~~~~~~~~~

.. figure:: /imgs/StatSTEMwithModel.png 
    
    The optimised model of an experimental high-angle ADF (HAADF) STEM image of a Pt/Ir sample, overlain with the fitted atomic column coordinates.

The standard way of modelling each atomic column in the image as a
Gaussian peak is to optimise the parameters of the provided atomic
columns indicated in the *Preparation* panel. The parameters of atomic
columns that are not initially indicated cannot be optimised by the
standard procedure. In order to start the fitting procedure, at least
one starting coordinate needs to be defined. Detailed information about
the fitting routine can be found in Ref. [@DeBacker2016]. By hitting the
*Run fitting routine* button, the fitting process is initiated. When the
routine is done, the optimised model is shown (see Figure
`4.1 <#fig:StatSTEMmodel>`__). The fitting procedure can be aborted by
hitting the *Abort fitting routine* button. When the procedure is
aborted, StatSTEM shows a message in the message bar at the bottom of
the GUI. Along with fitting the model, also the total intensity of
electrons scattered by each atomic column, the so-called scattering
cross-section, is calculated. These scattering cross-sections can be
visualised by a histogram, available at the right-hand side in *Select
Image* (see Figure `4.2 <#fig:StatSTEMhist>`__).

.. figure:: /imgs/StatSTEMwithHistogram.png 
    
    Histogram of scattering cross-sections of the atomic columns in an experimental HAADF STEM image of a Pt/Ir sample.

The fitting routine can be adjusted by changing its options which are
available by ticking the *Show options* box. In the following sections,
the different options are discussed.

Background
^^^^^^^^^^

Here, the user can choose to fit a constant background by ticking the
*Fit background* box. If the background is chosen not to be fitted, a
constant background value should be given. This value can be directly
entered or obtained from the average pixel values of a region of the
image that can be indicated by using the *Select* button.

Column width
^^^^^^^^^^^^

In the fitting procedure, there is the possibility to fit the Gaussian
peaks to the atomic columns with equal or different widths. In the *Same
width* option, the estimated Gaussian peaks have the same width for
columns of the same atom type, as labelled in the *Preparation* panel
discussed in section `3 <#chap:prep>`__. In the *Different width*
option, a different width is estimated for each Gaussian peak. In the
*User defined* option, the user can define a value for the width of each
column type (Å). By default, the *Same width* option is used as this is
computationally less demanding as compared to fitting all the peaks with
a different width and provides reliable results.

Test for convergence
^^^^^^^^^^^^^^^^^^^^

This option may be used for quickly testing the correctness of the
starting coordinates and fitting parameters. In this case, the number of
iterations is limited to 4. After a test is done, the newly obtained
coordinates may be preferred to use as starting coordinates. By hitting
the *Re-use fitted coordinates* button, the quickly fitted coordinates
can be used as new starting positions for the atomic columns.

Parallel computing
^^^^^^^^^^^^^^^^^^

For improving computational speed, the fitting procedure uses parallel
computing in which the calculations are divided over the different CPU
cores of the computer. The *Number of CPU cores* used for parallel
computing may be reduced to lower the CPU usage during the fitting
procedure. Be aware that then total calculation time possibly inceases.

Model Selection (MAP)
~~~~~~~~~~~~~~~~~~~~~

Here, a more general way of modelling electron microscopy images, as
compared to the standard procedure described in section
`4.1 <#sec:stan>`__, is offered. This method makes use of the concept of
model selection in combination with a Bayesian framework for detecting
the number of atomic columns present in the image data. For this, use is
made of the so-called maximum a posteriori (MAP) probability rule which
has been proposed for single atom detection [@Fatermans2018]. The
routine searches for possible column locations and compares the
probabilities of candidate models to each other. As such, the most
probable number of atomic columns and corresponding parametric model can
be automatically derived from the available image data. More details on
the working principles of this methodology can be found in Ref.
[@Fatermans2019].

General info
^^^^^^^^^^^^

Similarly as for the standard procedure of model fitting mentioned in
section `4.1 <#sec:stan>`__, the parametric model consists of Gaussian
peaks, describing the projected atomic columns, superposed on a constant
background. Here, there is no option though to change the values of the
background or column width. By default, a constant background and equal
column widths are fitted. Best results are obtained when the background
in the image can be adequately modelled as a constant. When this is not
the case, consider cropping the image (see section `3.4.2 <#sec:cut>`__)
so that the background can be sufficiently described to be constant. As
opposed to the standard procedure, it is optional for the
model-selection method to have starting coordinates provided in the
*Preparation* panel. When no initial coordinates are given, the routine
starts from fitting zero atomic columns (so only background) and
continues to search for more columns from there. When input has been
given, the routine starts from the provided input coordinates. Note
that, in order to apply the model-selection method, there is no use in
providing different column types for the input coordinates (see section
`3.3 <#sec:types>`__), since in the detection mechanism no
differentation can be made between different column types.

Electron counts
^^^^^^^^^^^^^^^

For correctly applying atom column detection by MAP probability, it is
crucial that the image data is properly converted to electron counts. As
mentioned earlier in section `3.4.5 <#sec:norm>`__, StatSTEM offers this
possibility when beam current and pixel dwell time are known and when
the image can be normalised by using a detector scan. There are also
alternative ways for converting image data to electron counts
[@Krause2016]. When images are recorded using direct-electron detectors,
conversion to electron counts might not be necessary anymore, as the
image is directly recorded in this format. **Always be cautious whether
conversion to electron counts is still needed when applying the MAP
methodology!** In case of doubt that the electron conversion was done
correctly, you can apply Poisson noise to the fitted model. This should
lead to a similar looking image as the original image with a similar
range of pixel intensities.

Running MAP
^^^^^^^^^^^

.. figure:: /imgs/OverviewMAP.png 
    
    Extra screen opening when MAP routine is finished and showing an overview of the fitted number of atomic columns and optimised models for HAADF STEM image data of SrTiO\ :subscript:`3`.

For optimal results, the model-selection (or atom detection) routine
should be applied to images exhibiting low SNR and low contrast, and
thus to low contrast-to-noise ratio (CNR) images. When there are regions
in the image where atomic columns are easily recognisable (for example
the middle section of a nanoparticle), it is better to avoid these areas
and focus on the more challenging sections only (for example the border
region of a nanoparticle). Cropping images can be done directly in
StatSTEM, as explained in section `3.4.2 <#sec:cut>`__. By hitting the
*Run model selection* button, the generalised fitting routine is
started. The routine can be aborted by pressing *Abort fitting routine*,
similarly as for the standard procedure. While the calculations are
being performed, progress can be followed through the progress bar at
the bottom right of the GUI and through the updating of the relative
probability curve and detected atom column coordinates shown on the raw
image data and the optimised model. When the procedure is finished, the
user is asked to indicate the desired number of columns (and associated
optimised model) from the relative probability curve for further
analysis in StatSTEM by using the cursor. For this, the user can rely on
an extra screen (*Overview MAP*) that opens up when the calculation is
finished (see Figure `4.3 <#fig:overviewmap>`__). Here, an overview is
given of the different models and corresponding number of atomic columns
that have been fitted during the procedure. By default, the most
probable number of atomic columns is initially highlighted. It is noted
that the extra screen is merely a helping tool for deciding what number
of atomic columns should be chosen. Closing this screen does not affect
StatSTEM. Once the desired number of atomic columns is chosen by using
the cursor from the relative probability curve in the main screen of
StatSTEM, the extra screen is automatically closed and an extra plot
option in *Select Image* at the right-hand side appears for visualising
the probability curve as a function of the number of atomic columns with
the chosen number of columns highlighted (see Figure
`4.4 <#fig:StatSTEMMAP>`__). If one is not satisfied with the selected
number of atomic columns, another number can be chosen through the
*Analysis* panel. More information on this follows in section
`5.1.1 <#sec:newmod>`__.

.. figure:: /imgs/StatSTEMMAP.png 

    Relative probability curve and indicated number of atomic columns by using MAP probability for an experimental HAADF STEM image of a SrTiO\ :subscript:`3` sample.

Options
^^^^^^^

Several settings are available when running the model-selection
procedure by the MAP probability rule. The most important option is the
*Max. # columns* setting. When this is specified to be 0 or any number
smaller than or equal to the number of provided starting coordinates,
specified in the *Preparation* panel (see sections `Peak
Finding <../../b_Preperation/Peak_Finder>`__ and `Add and remove
peaks <../../b_Preperation/Add_Peaks#sec:addrempeaks>`__), the
model-selection method continues until the probabilities of newly added
column coordinates drop. Thus, in this case, the routine is
automatically stopped when maximum probability is reached. Since it is
unknown when the routine will terminate exactly, the progress bar may
fluctuate. On the other hand, when the *Max. # columns* is put to a
number greater than the number of initial coordinates, the calculation
runs until the provided number of columns. The remaining options are
more advanced settings and provide the ranges between which the
parameters of the model (background, column width, column intensities
and column coordinates) are expected to lie a priori. More information
on this can be found in Ref. [@Fatermans2019]. By default, the ranges
for the background and column intensities are defined between 0 and the
maximum pixel intensity, whereas the values for the column width and
coordinates are defined within the field of view of the image. Note that
it is not possible to define individual ranges on column width,
intensity and position for different atomic columns.

Guidelines
^^^^^^^^^^

Sometimes, running MAP under the default options may not be optimal
which can lead to unrealistic model fits. For stabilising the fitting
routine in the MAP procedure, consider providing more physical values
for *Min. width* and *Max. width* related to a realistic range for the
typical width of an atomic column in the image. Also, when considering
small nanoparticles on a background, it is general good practice to
provide a value for *Min. intensity* which indicates the minimum
expected intensity of a single atom in the image. If this value is not
specified, MAP can become sensitive to small variations in the
background which leads to overdetection and the indication of possible
atomic columns which are physically unrealistic. A value for *Min.
intensity* can, for example, be determined from an image simulation of a
single atom. Typically, the ranges related to the positions of the
atomic columns should not be changed. In addition, MAP should not be
applied in a situation where background has been altered or removed from
an image (see section `3.4.4 <#sec:back>`__). The MAP procedure is
designed to recognise atomic columns from noise and removing background
hinders its judgement. For fitting with the standard procedure (see
section `4.1 <#sec:stan>`__), background removal can be applied. Lastly,
it should be stressed that the use of the MAP methodology in StatSTEM
should be applied to images containing a limited number of atomic
columns (<100). **Apply MAP only on images of small nanoparticles or
images of bulk materials with a limited field of view!**





