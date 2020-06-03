# Solitons
I used the python 3.6.4 program IDLE run on macOS Mojave version 10.14 for this project. Python is a very readable language and it is also very diverse in its functions especially for scientific programming. My coding style is more to allow what is being done to be clear to the user which is why I constantly use while loops due to the intuitiveness of counters, this is also the reason for me using nested lists. I prefer to have my plots next to where the data is produced for ease of understanding.

To see the animation of a single soliton, set animation = True.

Figure 1a. Set only plotsoliton = True in the beginning of the code, and enter parameters plotsol(2.0,2.1,0.5) in line 490. Set dx = 0.15. (Time~few seconds).

Figure 1b. Set only plotsoliton = True in the beginning of the code, and enter parameters (3.0,1,0.3). Set dx = 0.1. (Time~few seconds).

Figure 2a. Set only plotcmap = True, parameters (0.6,30). Set dx = 0.15. (Time~few seconds).

Figure 2b. Set only plotcmap = True, parameters (1.0,30). Set dx = 0.15. (Time~few seconds).

Figure 3. Set only velplot = True. Set dx = 0.05. (Time~1 minute).

Figure 4a. Set only stabilityplot = True in the beginning of the code, and enter parameters alpha = 0.05 in beginning of code. (Time~few minutes (< 10 minutes)).

Figure 4b. Set only stabilityplot = True in the beginning of the code, and enter parameters alpha = 0.5 in beginning of code. (Time~few minutes (< 10 minutes)).

Figure 4c. Set only stabilityplot = True in the beginning of the code, and enter parameters alpha = 9.0 in beginning of code. (Time~few minutes (< 10 minutes)).

Figure 5a. Set only plotcmapcol = True in the beginning of the code, and enter parameters umapcol(1.0,1.2,30). Set dx = 0.15. (Time~few seconds).

Figure 5b. Set only plotcmapcol = True in the beginning of the code, and enter parameters umapcol(1.0,2.2,30). Set dx = 0.15. (Time~few seconds).

Figure 6. Set only wave break = True in beginning of code. and enter parameters wavebreakplot(20,2). Set dx = 0.15. (Time~few seconds).

Figure 7. Set sine wave period 10 in s() s function, then run like figure 6.

Figure 8. Set only plotcmapwb = True, set alpha = 1.0,plotcmapwb(60). (Time~1 minute).

Figure 9a. Set only shockwave = True, set alpha = 1.0,  shockwaveplot(50,2). i += 5. (Time~1 minute).

Figure 9b. Set diff = True, plotdiff = False, set alpha = 1.0,  shockwaveplotdiff(40,0.2). i += 40. (Time~1 minute).

Figure 10. notdiff = True, arguments umap(1.0,40,np.arange(0,60,dx) in line 803. (Time~few seconds).

Figure 11. diff = plotdiff = True, arguments umap(1.0,60,np.arange(0,60,dx)) in line 796. (Time~few seconds).

