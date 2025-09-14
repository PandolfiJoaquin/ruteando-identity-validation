import os
import pandas as pd
from app import _verify 

RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
RESET = '\033[0m'

def get_test_cases():


    files = os.listdir("/home/joaco/Desktop/tutoria-piñeiro/ruteando/ruteando-validator/dataset")
    subjects = [f.split("_")[1] for f in files if "dni" in f]
    testcases = []
    for s1 in subjects:
        for s2 in subjects:
            testcases.append((f"cara_{s1}", f"dni_{s2}", "ACCEPT" if s1 == s2 else "REJECT"))

    return testcases 


def test_verifications():
    testcases = get_test_cases()
    results= []
    for s1, s2, expected in testcases:
        print("TEST", s1, s2, end='... \t\t')
        verification_result = _verify(s1, s2)
        color = choose_color(expected, verification_result)
        print(f"{color}{verification_result.scores}{RESET}" if verification_result.scores else "-")
        results.append((s1, s2, expected, verification_result.result, verification_result.scores, verification_result.reason))
    
    postprocess_results(results)

def choose_color(expected, verification_result):
    if verification_result.decision == "ACCEPT" and expected == "REJECT":
        return RED
    return GREEN if verification_result.decision == expected else YELLOW

def postprocess_results(results):
    df = pd.DataFrame(results, columns=["face", "dni", "expected", "predicted", "score", "reason"])
    df['correct'] = df['expected'] == df['predicted']
    df['is_false_positive'] = (df['expected'] == "REJECT") & (df['predicted'] == "ACCEPT")
    df.to_csv("test_results.csv")
    print(df.pivot_table(index='expected', columns='predicted', values="face", aggfunc='count'))



if __name__ == "__main__":
    test_verifications()
