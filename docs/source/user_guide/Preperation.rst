Preperation
-----------

In StatSTEM, model-based parameter fitting is applied for investigating
STEM images. The parametric model that is used consists of a
superposition of Gaussian peaks, describing the projected atomic column
intensities. For the parameter-optimisation procedure, starting
coordinates for the atomic column positions need to be defined.
Different options for choosing these initial coordinates adequately are
available in the *Preparation* panel.

Image Preperation
~~~~~~~~~~~~~~~~~

In this panel, general image parameters can be changed and image
operations can be executed.

Pixel size
^^^^^^^^^^

When an image is loaded to StatSTEM, the pixel size is standard put to
1 Å. This value can easily be changed, which results in a rescaling of
the starting coordinates and fitted parameters. **Note that it is
important to provide an accurate value for the image pixel size since
the reliability of certain calculations in StatSTEM depends on this
value!**

Cut part from image
^^^^^^^^^^^^^^^^^^^

This option allows one to cut out a rectangular region of interest from
the loaded image in order to perform the image quantification on a
smaller section. This is done by dragging the left-mouse button over the
image, selecting the area of interest. By clicking, the selected region
is confirmed and the image is cropped. By pressing the *escape* key, the
cropping process can be cancelled. Note that once the image cropping has
been performed, the original image cannot be retrieved. In order to
retrieve the original image, it needs to be reloaded in StatSTEM using
the *Load* button, as described in section `2.1.1 <#chap:load-save>`__.

Flip image contrast
^^^^^^^^^^^^^^^^^^^

Typically, atomic columns in annular dark-field (ADF) STEM images are
depicted as bright spots on a dark background. By hitting the *Flip
contrast image* button, the image contrast can be reversed, meaning that
the atomic columns are displayed as dark spots on a bright background.
This function can also be used to flip the contrast of annular
bright-field (ABF) STEM images.

Replace image background
^^^^^^^^^^^^^^^^^^^^^^^^

.. figure:: /imgs/ReplaceBack.png 

    Question popping up on how to replace the background value after hitting **Replace image background**.

By using this function, part of the image intensities can be replaced by
a certain background intensity. For example, this function may be useful
for removing image contributions from neighbouring nanoparticles. When
pressing *Replace image background*, the user is asked on how the new
background value will be provided, as shown in Figure
`3.2 <#fig:back>`__. When *Select region in current image* is selected,
an area needs to be indicated on the current figure by drawing its
corner points one by one. By right clicking, the last corner point is
automatically connected to the starting point. The new background value
is equal to the average of the pixel intensities within the drawn
region. Then, the region of the image to which the new background value
should be applied should be drawn in a similar way. The routine can be
aborted by pressing the *escape* key. When selecting *Select region in
other image*, the procedure works in the same way as described before,
but this time the new background value is selected by indicating an area
on another image. Lastly, by clicking *By value*, the new background
value needs to be provided as a number. Similarly as described in
section `3.4.2 <#sec:cut>`__ for cropping the image, once the background
has been replaced by another value, the original image can only be
retrieved by reloading it through the *Load* button at the bottom left
of the GUI.

Normalise image
^^^^^^^^^^^^^^^

.. figure:: /imgs/NormIm.png 
    
    Example showing typical values to normalise an image. Standard :math:`I_{vac}` and :math:`I_{det}` are put to 0 and 1, respectively.

Normalising STEM image data is important for directly comparing
experimental images with simulated ones. Typically, image normalisation
is achieved by a detector scan from which the averaged image intensity
in vacuum :math:`I_{vac}`, corresponding to the area next to the detector, and the averaged detector intensity (:math:`I_{det}`) can be measured. A normalised image (:math:`I_{norm}`) from the raw image (:math:`I_{raw}`) is then calculated by the following equation: It is important to stress that the raw image data and the detector scan should be recorded
under the same imaging conditions. More information on this topic can be
found in Ref. [@Krause2016]. When clicking the *Normalise image* button,
the user can change the *Intensity vacuum* (:math:`I_{vac}`) and *Intensity detector* (:math:`I_{det}`) values. Alternatively, these values can be directly loaded from a
detector scan by pressing *Load values from detector map*. When the
detector file consists of multiple detector scans, StatSTEM opens a
window where it can be indicated which detector scan should be used or
which scans should be averaged. When proper normalisation values have
been chosen, the normalisation is performed when *Ok* is selected. If
the box *Convert image to electron counts* is ticked on, the user can
specify the pixel dwell time (:math:`\mu s`) and beam current (pA) which allow to calculate the incidentelectron dose. In this case, pressing *Ok* converts the raw image into
electron counts. This conversion is necessary for correctly applying the
maximum a posteriori (MAP) probability rule for atom column detection,
which is further explained in section `4.2 <#sec:map>`__.

Get peak locations
~~~~~~~~~~~~~~~~~~

Peak-finder routine
^^^^^^^^^^^^^^^^^^^

In StatSTEM, two peak-finder routines are available (*Peak-finder
routine 1* and *Peak-finder routine 2*) which search for local maxima in
the image. When using these peak finders, a new window opens where the
parameters of the peak-finder routine can be tuned. These routines work
by using filters for smoothing the image. In *Peak-finder routine 1*,
there is the option of adding three different filters: an *average*,
*disk*, and *gaussian* filter. In *Peak-finder routine 2*, there is no
explicit way of altering the filter, but this is implicitly achievable
by altering the *Estimated Radius* of the atomic columns. For both
peak-finder routines, a threshold value can be defined for removing
nuisance pixel intensities from the background. Lastly, *Peak-finder
routine 2* offers an extra option, which is defining the *Minimum
Distance* between the projected atomic columns in the image. In the
peak-finder window, the routine can be tested for a variety of settings.
When satisfactory peak locations are found, these coordinates can be
exported to StatSTEM by hitting *Use values* or *Confirm values* for
*Peak-finder routine 1* or *Peak-finder routine 2*, respectively.

.. figure:: /imgs/Peakfinders.png

    Examples of applying\ **Peak-finder routine 1**\ (top), and\ **Peak-finder routine 2**\ (bottom) to an image of graphene.

Import locations from file
^^^^^^^^^^^^^^^^^^^^^^^^^^

Atom column coordinates can also be loaded manually into StatSTEM by
clicking *Import locations from file*. For this, a MATLAB (*.mat*) or
text (*.txt*) file needs to be used with the x-coordinates in the first
column and the y-coordinates in the second column expressed in Ångström.
It is noted that when a previous analysis of StatSTEM is loaded through
the *Load* button (see section `2.1.1 <#chap:load-save>`__), also the
starting coordinates of this analysis are loaded and can be used as
initial coordinates for the new analysis as well.

Get locations from MAP
^^^^^^^^^^^^^^^^^^^^^^

By pressing the *Get locations from MAP* button, the coordinates of the
atomic columns detected by the maximum a posteriori (MAP) probability
rule are loaded. This option is only available when the MAP rule has run
previously in StatSTEM, or when the loaded StatSTEM file contains an
outputStatSTEM_MAP class. More details on the use and working principle
of the MAP rule for atom column detection follows in section
`4.2 <#sec:map>`__.


Add/Remove peaks
~~~~~~~~~~~~~~~~

In this panel, different routines are available to define, remove, or
change starting coordinates manually.

Type
^^^^

With this option, it is possible to specify the types of atomic columns
that are present in the image. For example, it can be used when the
image consists of atomic columns of different elements. In the drop-down
menu, you can choose to *Add* or *Remove*, or specify the *Names* of the
atom types, which are otherwise labelled as numbers. In section
`3.3 <#sec:types>`__, further options are discussed for handling
different atom types.

Add and Remove
^^^^^^^^^^^^^^

By clicking the *Add* button, you can manually indicate the positions of
the atomic columns in the image. By pressing the *escape* key or
clicking outside the image, you exit the peak-adding mode. The types of
the added peaks need to be specified by the drop-down menu left from the
*Add* button. Coordinates labelled by different types are shown in
different colours. Similarly, you can also manually remove peaks by
clicking the *Remove* button. This allows you to remove individual
coordinates from the image. For removing all coordinates, you can simply
press *Remove all*.


Select and Remove region
^^^^^^^^^^^^^^^^^^^^^^^^

By pressing *Select region*, the user can select a region from the image
in which the starting coordinates should be maintained. Outside the
selected region, all starting coordinates are removed. In this routine,
the user defines the corner points of the selected area one by one.
Right clicking connects the last defined point with the starting point.
By pressing the *escape* key, the routine is aborted. The button *Remove
region* works analogously, but the opposite happens. Here, a region from
the image is selected from which the starting coordinates should be
removed. The coordinates outside this region are maintained.


Assign column types
^^^^^^^^^^^^^^^^^^^

In this panel, different options are listed to deal with an image where
different atomic column types are present.

Projected unit cell
^^^^^^^^^^^^^^^^^^^

In StatSTEM, an automatic routine is available that can identify the
different column types in an image from a given projected unit cell.
Here, the relative location of each column, together with the lattice
parameters, should be given. You can use the buttons *New* and *Delete*
to add and remove an atomic column, respectively. All columns, except
the first one, can be removed by using the button *Clear*. The user can
also provide information on the depth location of each atom in a column
by using the button *z-information*. Its functionality is explained in
section `5.2.1 <#strain:PUC>`__. Make sure that in StatSTEM you fill in
the correct pixel size of the image (see section `3.4.1 <#sec:pix>`__),
as the lattice parameters in the projected unit cell should be close to
the experimental values. It is noted that StatSTEM contains a database
with projected unit cells for some common materials and viewing
directions.

.. figure:: /imgs/ProjUnit.png 
    
    The atomic column locations in a projected unit cell of Au, viewed along the [100]-direction.

Auto assign
^^^^^^^^^^^

By using this tool, StatSTEM identifies different column types which are
present in the image. In this procedure, lattice directions are first
determined by comparing the most central coordinate with its
neighbouring coordinates. For this, input on the projected unit cell is
required (see section `3.3.1 <#input:PUC>`__). Then, columns are indexed
with respect to the central coordinate. In this manner, the positions of
all columns in the projected unit cell are identified and different
column types can be assigned.

Add missing types
^^^^^^^^^^^^^^^^^

In this procedure, StatSTEM uses the projected unit cell to find and add
the locations of missing column types, following a similar procedure as
for *Auto assign*.

Change type to
^^^^^^^^^^^^^^

With this function, a region in the image can be selected where all the
contained atom column types are changed to the selected type label,
indicated by the drop-down menu to the right of the *Change type to*
button. The routine works in the same way as described in section
`3.2.3 <#sec:sel-rem>`__ where the user defines the corner points of the
selected area one by one. Right clicking connects the last defined point
with the starting point. By pressing the *escape* key, the routine is
aborted.





