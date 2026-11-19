# Generate docs/assets/demo.gif in ~10 seconds with Charmbracelet VHS:
#   vhs docs/assets/demo.tape

Output docs/assets/demo.gif

Require osprey

Set Shell "bash"
Set FontFamily "JetBrains Mono"
Set FontSize 14
Set Width 1280
Set Height 760
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Type "osprey --version"
Enter
Sleep 800ms

Type "osprey doctor --bench-mb 16"
Enter
Sleep 2s

Type "osprey bench --vectors 5000 --dim 384 --queries 20 --clusters 32 --nprobe 8 --candidates 64"
Enter
Sleep 4s
