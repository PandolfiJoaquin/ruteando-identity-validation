import os
import pandas as pd
from app import _verify, pil_from_local_storage


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
        img1 = pil_from_local_storage(s1)
        img2 = pil_from_local_storage(s2)
        print("validating for: ", s1, s2)
        verification_result = _verify(img1, img2)
        results.append((s1, s2, expected, verification_result.decision, verification_result.score))
    
    postprocess_results(results)

def postprocess_results(results):
    df = pd.DataFrame(results, columns=["face", "dni", "expected", "predicted", "score"])
    df['correct'] = df['expected'] == df['predicted']
    df['is_false_positive'] = (df['expected'] == "REJECT") & (df['predicted'] == "ACCEPT")
    df.to_csv("test_results.csv")
    print(df.groupby('is_false_positive').count())




if __name__ == "__main__":
    test_verifications()