#!/usr/bin/python3

script_info="""
Make a simulated CTF curve and lets you play around with the various parameters. Early version by Zuben P. Brown, Frank Lab
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse

#This is just a simple parser so I can change some of the variables
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=script_info)
parser.add_argument(
                   '-k', 
                   '--spatial_frequency', 
                   default=(0.5), 
                   type=float,help='The limit of the spatial frequency',
                   required=False
                    )
parser.add_argument(
                   '-kV',
                   '--voltage',
                   default=(300),
                   type=int,
                   help='In kV.',
                   required=False
                    )
parser.add_argument(
                   '-A',
                   '--amp_contrast',
                   default=(0.1),
                   type=float,
                   help='If we want to change the amplitude contrast',
                   required=False
                    )
parser.add_argument(
                   '-B',
                   '--envelope_function',
                   default=(50),
                   type=float,
                   help='Sets the B-factor',
                   required=False
                    )
parser.add_argument(
                   '-Cs',
                   '--Spherical_aberration',
                   default=(2.0),
                   type=float,
                   help="The Cs in mm of the EM. Will be converted from mm to Angstroms automatically"
                    )
parser.add_argument(
                    '-def', 
                    '--defocus',
                    default=(1),
                    type=float,
                    help='The defocus value in micrometers. Positive for underfocus after Heymann et al. (2005)',
                    required=False
                    )
parser.add_argument(
                    '-s',
                    '--save',
                    nargs='?',
                    const='Test',
                    default=False,
                    required=False,
                    help='Saves an image with the input as the file name, probably works best if you use underbar instead of spaces'
                    )
parser.add_argument(
                    '-p',
                    '--perfect_CTF',
                    action='store_true',
                    required=False,
                    help='adds the perfect CTF to the output'
                    )
parser.add_argument(
                    '-z',
                    '--Zhu',
                    required=False,
                    action='store_true',
                    help='Uses the CTF equations from Zhu et al. 1997. Normally I use the equations from Zhang 2016, even though they are equivalent (except for the 2 exponent in Zhu et al. 1997 which I ignore anyway)'
                    )
parser.add_argument(
                    '--hide',
                    required=False,
                    action='store_true',
                    default=False,
                    help='Hides the envelope function and limited CTF. Will show only perfect CTF'
                    )
parser.add_argument(
                    '-f',
                    '--flip',
                    required=False,
                    action='store_true',
                    help='Do you liked flipped phases? Also tells you the area under the curve'
                    )
parser.add_argument(
                    '-S',
                    '--Scherzer',
                    required=False,
                    action='store_true',
                    help='Sets the defocus as the Scherzer defocus, which maximises the area under the first peak. '
                    )
args=parser.parse_args()
kvoltage=(args.voltage)*1000                # Converts the kV into voltage
max_spatial_freq=args.spatial_frequency     # Something to set where the maximum spatial frequency will go to
amplitude_contrast=args.amp_contrast        # Allows me to play around with the amp contrast
B_factor=args.envelope_function             # Sets the envelope B-factor
Cs=(args.Spherical_aberration)*1000*1000*10 # The Cs. Converts from mm to Angstroms
defocus=args.defocus*1000*10                # Gets the defocus (input as micrometers so convert to Angstroms).
save=args.save
Zhu=args.Zhu                                # Use the equations from Zhu et al. 1997 instead
flip=args.flip                              # Flip phases, and reports area under curve
hide=args.hide
k=np.arange(0,max_spatial_freq,0.0001)      # Maximum spatial frequency 

#relativistic_lambda=12*(voltage_power)
relativistic_lambda=12.2643247 / np.sqrt(kvoltage * (1 + kvoltage * 0.978466e-6))

Scherzer_defocus=1.2*np.sqrt(Cs*relativistic_lambda)
if args.Scherzer==True:
    defocus=Scherzer_defocus
else:
    pass
print("""
The Scherzer defocus for these parameters is: {0:.2f} Angstroms
""".format(
            Scherzer_defocus 
            )
    )


fig= plt.figure()
ax = fig.add_subplot(111)

if Zhu == True:
    gamma=(2*np.pi)*(-0.5*defocus*relativistic_lambda*np.power(k,2)+0.25*Cs*np.power(relativistic_lambda,3)*np.power(k,4))        # From Zhu et al. 1997
    CTF=2*(np.sin(gamma)- amplitude_contrast*np.cos(gamma))                                             # From Zhu et al. 1997 *** scaled to 1 not 2 as in paper
else:
    gamma=(-np.pi/2)*Cs*np.power(relativistic_lambda,3)*np.power(k,4)+np.pi*relativistic_lambda*defocus*np.power(k,2)             # Zhang 2016
    CTF = - np.sqrt(1-np.power(amplitude_contrast,2))*np.sin(gamma) - amplitude_contrast*np.cos(gamma)  # This is from Zhang 2016; CTF=-sqrt(1-A^2)Sin(gamma) - A cos(gamma)

k_squared=np.power(k, 2)
E=np.exp(-1*B_factor*k_squared) # This is the generalised B-factor from Frank, 2006
Y=CTF*E

# This will flip the phases so that they are all positive
if flip==True:
    Y=np.abs(Y)
    print("The area under the curve is roughly: {0:.2f} units squared".format(np.trapz(Y)))
else:
    pass

if hide==False:
    plt.plot(      # This plots the CTF closed by the envelope function
        k, 
        Y,
        color="k")
    plt.plot(       # Upper envelope fuction
        k,
        E, #np.exp(-1*B_factor*k_squared),
        color="red")
    plt.plot(       # Lower envelope function
        k,
        -E,
        color="red")
else:
    pass

if args.perfect_CTF==True:
    plt.plot(       # The 'perfect' CTF with no envelope function. Show with [-p]
        k,
        CTF,
        color="b")
else:
    pass

#plt.legend(title="B-factor")
plt.title("CTF for: \n {0}kV, A={1}, B={2}, Cs={3}, {4} defocus".format(
                                                                        args.voltage,
                                                                        amplitude_contrast,
                                                                        B_factor,
                                                                        args.Spherical_aberration,
                                                                        defocus/10000
                                                                        )
         )

#ax.axis([min(k),max(k),-1,1])
ax.set_xlabel("Spatial frequency (k)", fontsize=15)
ax.set_ylabel("Contrast", fontsize=15)
ax.tick_params(axis="both", labelsize=15)
plt.grid()

print(
"""
The settings used are:
{0} kV
{1} amplitude contrast
{2} B-factor
{3} Cs
{4} Defocus (micrometers)
""".format(
            kvoltage/1000,          # Convert to kV
            amplitude_contrast,     # Fraction
            B_factor,               # Angstrom's squared
            Cs/10000000,            # Convert to mm from Angstroms
            defocus/10000           # Convert to um from Angstroms
            )
    )
if args.save:
    plt.savefig("{0}_{1}kV_{2}A_{3}B_{4}Cs_{5}defocus.png".format(
                                                                    save,
                                                                    args.voltage,
                                                                    amplitude_contrast,B_factor,
                                                                    args.Spherical_aberration,
                                                                    defocus/10000)
                                                                    )
else:
    plt.show()
