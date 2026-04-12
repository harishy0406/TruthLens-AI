"""Prediction test with realistic news-style text."""
import sys
sys.path.insert(0, '.')
from src.utils.predict import predict

tests = [
    ('FAKE', """
    BREAKING: Scientists confirm that drinking bleach mixed with lemon juice cures COVID-19 in 24 hours.
    A secret government study suppressed by mainstream media reveals this miracle cure that Big Pharma
    doesnt want you to know about. Share this before it gets deleted by the deep state!
    According to insiders the WHO and CDC are hiding this from the public to protect vaccine profits.
    """),
    ('REAL', """
    WASHINGTON (Reuters) - The Federal Reserve held interest rates steady on Wednesday, as policymakers
    grappled with stubborn inflation and signs of slowing economic growth. Fed Chair Jerome Powell said at
    a press conference that the committee would remain data-dependent and would adjust policy as needed
    based on incoming economic information. Markets had widely expected the decision, with futures traders
    pricing in a roughly 95% chance of no change before the meeting.
    """),
    ('REAL', """
    CAPE CANAVERAL, Fla. (AP) - NASA's James Webb Space Telescope has captured stunning new images of
    the Carina Nebula, revealing thousands of never-before-seen young stars in a star-forming region.
    The images were released Tuesday by NASA, the European Space Agency and the Canadian Space Agency.
    Scientists said the observations will help them better understand how planetary systems form and evolve.
    """),
    ('FAKE', """
    EXCLUSIVE: Bill Gates secretly admitted in leaked documents that COVID vaccines contain nanochips
    designed to track and control the global population through 5G towers. The internal Microsoft memo,
    obtained by whistleblowers, reveals that the New World Order has been planning this depopulation
    agenda since 2015. Spread this truth before the globalists shut it down!
    """),
    ('FAKE', """
    CNN, the fake news network, has once again been caught fabricating stories about President Trump.
    Deep state operatives embedded in the mainstream media are working to destroy America. Patriots
    need to wake up and realize that everything you see on TV is a lie designed to keep you enslaved.
    The true patriots are fighting back against this evil agenda!
    """),
    ('REAL', """
    LONDON (Reuters) - British Prime Minister announced new economic measures to address rising living
    costs on Thursday, including a package of support worth several billion pounds for households.
    The Treasury said the measures would be funded through a windfall tax on energy company profits,
    which have soared due to global commodity price increases following the conflict in Ukraine.
    """),
]

print("="*90)
print(f"  {'Expected':6}  {'Verdict':6}  {'Conf':6}  {'FakeScore':9}  Result")
print("="*90)
correct = 0
for expected, text in tests:
    r = predict(text.strip())
    v = r['verdict']
    match = 'CORRECT' if v == expected else 'WRONG  '
    correct += int(v == expected)
    icon = 'OK' if v == expected else '!!'
    print(f"[{icon}] [{expected:4}] -> {v:4}  {r['confidence']:.3f}  {r['fake_score']:.5f}   {match}")

print(f"\nAccuracy: {correct}/{len(tests)} ({correct/len(tests)*100:.0f}%)")
