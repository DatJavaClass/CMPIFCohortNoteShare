<p align="center"><img width="651" height="701" alt="ChatGPT Image Jun 11, 2026, 12_49_40 PM" src="https://github.com/user-attachments/assets/99da5453-5993-4993-8aac-51c0a4b6510c" /></p>

# Tools

Small utilities I built for the cohort, gathered here so they are easy to find. Each tool is its own repository; click any of them to reach its full README, downloads, and source.

## [CMPIF2100_LAB_VER_CHECK](https://github.com/DatJavaClass/CMPIF2100_LAB_VER_CHECK)
Two lines of script that save your sanity. If you run an instructor's code exactly as written and your output does not match the video, the usual cause is not a bug on your end, it is that you and the instructor are on different Python versions. Drop `import sys` then `print(sys.version)` into your first cell to see which version you are actually running. Nothing is broken, and you are awesome.

## [CMPIF2100-Lab-Transcriber-2.0](https://github.com/DatJavaClass/CMPIF2100-Lab-Transcriber-2.0)
Record your computer's audio and turn it into a text transcript, live, on your own machine. Point it at a Zoom or Teams lecture, a recorded class, or anything playing through your speakers, press **Record**, and watch the words appear as they are spoken. On **Stop** it saves both the `.wav` recording and a clean `.txt` transcript. **Windows only** (it captures system audio through WASAPI loopback); it runs on an NVIDIA GPU when one is available, otherwise the CPU. Version 2.0 records *and* transcribes in one program, no second tool needed. A portable `.exe` is available, or you can run it from source.

## [VEShell](https://github.com/DatJavaClass/VEShell)
*Very Easy Shell*: a reliable PowerShell terminal with rock-solid mouse and keyboard cut / copy / paste, built on xterm.js (the same engine as VS Code's integrated terminal), so it does not depend on conhost's flaky "Quick Edit" mode. It also bundles two learning tools: **ClaudeWhat** (highlight anything on screen and get a short, context-aware explanation of *that* selection) and **Verbose run** (describe a task in plain language and watch it happen one critical step at a time). An ongoing project, aimed at adding quality-of-life around the shell without losing CLI speed.

## [Linear Regression Simulator](https://www.datjavaclass.com/lrs/)
See what a line of best fit actually does, before the course formally reaches regression. Upload or paste a CSV (or type X and Y by hand), prune your columns down to the two you want to compare, and watch the scatter with fitted line, residuals, Q-Q plot, and histogram update together. Everything runs in your browser and nothing is uploaded. It is **live and ready to use** at the link above, no install needed. If you would rather keep or read the source, grab `DatJavaClassLRS.zip` in this folder; it ships with a sample dataset and an embedded README.

---

*About the old transcriber: version 1 (record-only, transcribe-with-a-separate-program) has been retired and archived. Use 2.0 above, it does the whole job in one place.*

*Browsing note: the folders above are Git submodules, so clicking one takes you straight to its own repo.*
