"""Co-registration subpackage.

This package wraps elastix / transformix execution for rigid (or later affine / deformable) alignment between imaging volumes.

Sub-modules
------------
registration_wrapper -> run()  : fire elastix with a supplied parameter file 
evaluation            -> Dice / Jaccard / MSE utilities when ground-truth masks exist 
visualize             -> simple overlay (matplotlib) helper
""" 