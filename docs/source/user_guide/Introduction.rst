Introduction
------------

.. figure:: /imgs/Overview_GUI.png
   
   Overview of the GUI of StatSTEM, highlighting the main sections.

The StatSTEM software package is aimed on providing a framework for
quantifying scanning transmission electron microscopy (STEM) images. For
this, StatSTEM makes use of model-based parameter fitting and
statistical techniques, providing accurate and precise quantitative
measurements. The quantification process proceeds in three steps:

-  Preparation (see section
   `Preperation <tutorial/Preperation/Preperation>`__)

-  Model fitting (see section `Model
   Fitting <tutorial/Fitting/Model_Fitting>`__)

-  Analysis (see section `Analysis <tutorial/Analysis/Analysis>`__)

The StatSTEM software package can be downloaded from the `StatSTEM
website <https://github.com/quantitativeTEM/StatSTEM>`__. The program is
started by opening MATLAB and running the main script *StatSTEM.m*,
entering the graphical user interface (GUI).

Overview
~~~~~~~~

.. figure:: /imgs/GUI_Load-Save.png
   
   Buttons for loading and saving files indicated by the green and red circles, respectively.

The main sections of the GUI of StatSTEM. The three steps of the
image-quantification process are displayed as separate tab panels in the
upper corner on the left-hand side. Each of these panels and their
underlying functions are thoroughly discussed further in this manual.
The right-hand side of the GUI keeps track of images and results that
are generated during the different steps of the quantification process.
At the bottom left of the GUI, the user has the possibility to load and
store files. Next to this, an information screen is located which
contains messages indicating when computations are finished, when
changes are made, or when an error occurred. At the bottom right, there
is a progress bar indicating the status of the current computation. As
computations are often iterative, it is noted that the progress bar is
not always a reliable indicator for estimating how long a computation
will take.

Load and save files
^^^^^^^^^^^^^^^^^^^^

.. figure:: /imgs/GUI_ImageOptions.png
   
   On the right-hand side of the GUI of StatSTEM, the desired image or plot can be displayed from the*\ Select Image\* panel, in combination with the indicated desired parameters from *Image Options* (both indicated by the red arrows). The *Export* button, indicated by the red circle, can be used to open the image in a new MATLAB figure.\*

The first step of using StatSTEM consists of loading the file you want
to investigate into the GUI. You can do this by clicking on the *Load*
button in the bottom left corner or on the addition tab in the top, as
indicated by the green circles in Figure `2.2 <#fig:load-save>`__. The
loaded file may be a single image or a by StatSTEM previously analysed
dataset. The preferred file format is that of MATLAB (*.mat*). When you
are satisfied with the analysis of your image data, you can save all the
generated results in the MATLAB format by clicking on the *Save* button
next to the *Load* button (see red circle in Figure
`2.2 <#fig:load-save>`__). In the saved file, all variables are stored
as StatSTEM class files. When working with these files, make sure that
the StatSTEM folder is always loaded to the path in MATLAB. In the class
files, a description is given on what information the parameters hold
and what their purpose is.

Select image
^^^^^^^^^^^^

In StatSTEM, the generated images and parameters during the
quantification process can easily be displayed. The *Select Image* panel
at the right-hand side allows to select the image or plot of interest.
Hereby, the different parameters in *Image Options* below *Select Image*
can be ticked off or on. By hitting the *Export* button, the currently
displayed image is opened in a new MATLAB figure which can then be saved
in the desired format.

Closing StatSTEM
^^^^^^^^^^^^^^^^

StatSTEM can be closed by hitting the cross in the top-right corner.
When closing StatSTEM, make sure that all results are saved, as no
warnings of unsaved results are given.





