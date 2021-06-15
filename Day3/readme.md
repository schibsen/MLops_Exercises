# 3. Debugging, profiling and visualizing code

## Debugging

Debugging is very hard to teach and is one of the skills that just comes with experience. That said, you should
familiarize yourself with the build-in [python debugger](https://docs.python.org/3/library/pdb.html) as it may come in
handy during the course. 

<p align="center">
  <img src="../figures/debug.jpg" width="700" title="hover text">
</p>

To invoke the build in python debugger you can either:
* If you are using an editor, then you can insert inline breakpoints (in VS code this can be done by pressing F9)
  and then execute the script in debug mode (inline breakpoints can often be seen as small red dots to the left of
  your code). The editor should then offer some interface to allow you step through your code.

* Set a trace directly with the python debugger by calling
  ```python
  import pdb
  pdb.set_trace()
  ```
  anywhere you want to stop the code. Then you can use different commands (see the `python_debugger_cheatsheet.pdf`)
  to step through the code.

### Exercises

We here provide a script `mnist_vae_bugs.py` which contains a number of bugs to get it running. Start by going over
the script and try to understand what is going on. Hereafter, try to get it running by solving the bugs. The following 
bugs exist in the script:

* One device bug (will only show if running on gpu, but try to find it anyways)
* One shape bug 
* One math bug 
* One training bug

Some of the bugs prevents the script from even running, while some of them influences the training dynamics.
Try to find them all. We also provide a working version called `vae_mnist_working.py` (but please try to find
the bugs before looking at the script). Successfully debugging and running the script should produce three files: 
`orig_data.png`, `reconstructions.png`, `generated_samples.png`. 

## Profilers

Using profilers can help you find bottlenecks in your code. In this exercise we will look at two different
profilers, with the first one being the [cProfile](https://docs.python.org/3/library/profile.html), pythons
build in profiler.

### Exercises

1. Run the `cProfile` on the `vae_mnist_working.py` script. Hint: you can directly call the profiler on a
   script using the `-m` arg
   `python -m cProfile [-o output_file] [-s sort_order] (-m module | myscript.py) `
   
2. Try looking at the output of the profiling. Can you figure out which function took the longest to run?

3. To get a better feeling of the profiled result we can try to visualize it. Python does not
   provide a native solution, but open-source solutions such as [snakeviz](https://jiffyclub.github.io/snakeviz/)
   exist. Try installing snakeviz and load a profiled run into it (HINT: snakeviz expect the run to have the file
   format `.prof`).

4. Try optimizing the run! (Hint: The data is not stored as torch tensor)

### Exercises (optional)

In addition to using pythons build-in profiler we will also investigate the profiler that is build into PyTorch already.
Note that these exercises requires that you have PyTorch v1.8.1 installed. You can always check which version you
currently have installed by writing (in python):

```python
import torch
print(torch.__version__)
```

Additionally, it PyTorch needs to be build with Kineto. This mean that if you get the following error when
trying to do the exercises:
```
Requested Kineto profiling but Kineto is not available, make sure PyTorch is built with USE_KINETO=1
```
You will sadly not be able to complete them. However, if not, the exercise will also require you to have the 
tensorboard profiler plugin installed:
``` 
pip install torch_tb_profiler
```

For this exercise we have provided the solution in form of the script `vae_mnist_pytorch_profiler.py` where
we have already implemented the PyTorch profiler in the script. However, try to solve the exercise yourself!

1. The documentation on the new profiler is sparse but take a look at this
   [blogpost](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/)
   and the [documentation](https://pytorch.org/docs/stable/profiler.html) which should give you an idea of 
   how to use the PyTorch profiler.

2. Secondly try to implement the profile in the `vae_mnist_working.py` script from the debugging exercises 
   (HINT: run the script with `epochs = 1`) and run the script with the profiler on.
   
3. Try loading the saved profiling run in tensorboard by writing
   ```
   tensorboard --logdir=./log  
   ```
   in a terminal. Inspect the results in the `pytorch_profiler` tab. What is the most computational expensive
   part of the code? What does the profiler recommend that you improve? Can you improve something in the code?

3. Apply the profiler to your own MNIST code.

### Experiment visualizers

While logging loss values to terminal, or plotting training curves in matplotlib may be enough doing smaller experiment,
there is no way around using a proper experiment tracker and visualizer when doing large scale experiments.

For these exercises we will initially be looking at incorporating [tensorboard](https://www.tensorflow.org/tensorboard) into our code, 
as it comes with native support in PyTorch

1. Install tensorboard (does not require you to install tensorflow)
   ```pip install tensorboard```

2. Take a look at this [tutorial](https://pytorch.org/docs/stable/tensorboard.html)

3. Implement the summarywriter in your training script from the last session. The summarywrite should log both
   a scalar (`writer.add_scalar`) (atleast one) and a histogram (`writer.add_histogram`). Additionally, try log
   the computational graph (`writer.add_graph`).
   
4. Start tensorboard in a terminal
   ```tensorboard --logdir this/is/the/dir/tensorboard/logged/to```
   
5. Inspect what was logged in tensorboard

Experiment visualizers are especially useful for comparing values across training runs. Multiple runs often
stems from playing around with the hyperparameters of your model.

6. In your training script make sure the hyperparameters are saved to tensorboard (`writer.add_hparams`)

7. Run at least two models with different hyperparameters, open them both at the same time in tensorboard
   Hint: to open multiple experiments in the same tensorboard they either have to share a root folder e.g.
   `experiments/experiment_1` and `experiments/experiment_2` you can start tensorboard as
   ```tensorboard --logdir experiments```
   or as
   ```tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2```

While tensorboard is a great logger for many things, more advanced loggers may be more suitable. For the remaining 
of the exercises we will try to look at the [wandb](https://wandb.ai/site) logger. The great benefit of using wandb
over tensorboard is that it was build with colllaboration in mind (whereas tensorboard somewhat got it along the
way).

1. Start by creating an account at [wandb](https://wandb.ai/site). I recommend using your github account but feel
   free to choose what you want. When you are logged in you should get an API key of length 40. Copy this for later
   use (HINT: if you forgot to copy the API key, you can find it under settings).

2. Next install wandb on your laptop
   ```
   pip install wandb
   ```

3. Now connect to your wandb account
   ```
   wandb login
   ```
   you will be asked to provide the 40 length API key. The connection should be remain open to the wandb server
   even when you close the terminal, such that you do not have to login each time. If using `wandb` in a notebook 
   you need to manually close the connection using `wandb.finish()`.

4. With it all setup we are now ready to incorporate `wandb` into our code. The interface is fairly simple, and
   this [guide](https://docs.wandb.ai/guides/integrations/pytorch) should give enough hints to get you through
   the exercise. (HINT: the two methods you need to call are `wandb.init` and `wandb.log`). To start with, logging
   the training loss of your model will be enough.

5. After running your model, checkout the webpage. Hopefully you should be able to see at least 

6. Now log something else than scalar values. This could be a image, a histogram or a matplotlib figure. In all
   cases the logging is still going to use `wandb.log` but you need extra calls to `wandb.Image` ect. depending
   on what you choose to log.

7. Finally, lets create a report that you can share. Click the **Create report** button where you choose the *blank*
   option. Then choose to include everything in the report 

8. To make sure that you have completed todays exercises, make the report shareable by clicking the *Share* button
   and create *view-only-link*. Send the link to my email `nsde@dtu.dk`, so I can checkout your awesome work.

9. Feel free to experiment more with `wandb` as it is a great tool for logging, organizing and sharing experiments.
