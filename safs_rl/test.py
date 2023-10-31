import jax
import flax
import numpy
import optax
import distrax
import matplotlib
import hydra
import brax
import jaxlib
import inflection
if inflection.__version__ =="0.5.1":
    print("inflection correct")
else:
    print(inflection.__version__, " != 0.5.1")
if jax.__version__ == "0.4.7":
    print("jax correct")
else:
    print(jax.__version__, " != 0.4.7")
if flax.__version__ == "0.7.2":
    print("flax correct")
else:
    print(flax.__version__ ,"!= 0.7.2")
if numpy.__version__ == "1.22.4":
    print("numpy correct")
else:
    print(numpy.__version__ ,"!= 1.22.4")
if optax.__version__ == "0.1.7":
    print("optax correct")
else:
    print(optax.__version__ ,"!= 0.1.7")
if distrax.__version__ == "0.1.3":
    print("distrax correct")
else:
    print(distrax.__version__ ,"!= 0.1.3")
if matplotlib.__version__ == "3.7.3":
    print("matplotlib correct")
else:
    print(matplotlib.__version__ ,"!= 3.7.3")
if hydra.__version__ == "1.3.2":
    print("hydra correct")
else:
    print(hydra.__version__ ,"!= 1.3.2")
if brax.__version__ == "0.9.2":
    print("brax correct")
else:
    print(brax.__version__ ,"!= 0.9.2")
if jaxlib.version.__version__ == "0.4.7":
    print("jaxlib correct")
else:
    print(jaxlib.version.__version__ ,"!= 0.4.7")
    
