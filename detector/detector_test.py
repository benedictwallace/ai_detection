from detector import Detector
d = Detector()

# Human-written text
human = [
    "The game pits two teams, Terrorists and Counter-Terrorists, against each other in different objective-based game modes. The most common game modes involve the Terrorists planting a bomb while Counter-Terrorists attempt to stop them, or Counter-Terrorists attempting to rescue hostages that the Terrorists have captured. There are nine official game modes, all of which have distinct characteristics specific to that mode. The game also has matchmaking support that allows players to play on dedicated Valve servers, in addition to community-hosted servers with custom maps and game modes. A battle-royale game-mode, Danger Zone, was introduced in late 2018.",
    "My cat's become really needy and wants strokes and affection all the time.",
    "After university, it's difficult to decide what the next steps should be.",
]

# Obviously AI-written text  
ai = [
    "The utilization of artificial intelligence in modern healthcare systems has demonstrated significant improvements.",
    "Furthermore, it is important to note that the aforementioned considerations are paramount.",
    "In conclusion, the implementation of these strategies will yield optimal outcomes.",
]

print("Human text:")
for t in human:
    print(f"  {d.score(t):.4f}  {t[:60]}")

print("\nAI text:")
for t in ai:
    print(f"  {d.score(t):.4f}  {t[:60]}")