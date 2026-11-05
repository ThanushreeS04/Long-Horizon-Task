class Auditor:
    # Verifies the expected environment fact instead of trusting a success message alone.
    def verify(self, step, result, environment):
        expected = step["expected"]
        observed = bool(environment.facts.get(expected, False))
        matches = result.get("fact") == expected
        verified = observed and matches and bool(result.get("evidence", {}).get("observed"))
        return {"verified": verified,
                "evidence": {"expected_fact": expected, "environment_fact_present": observed,
                             "result_matches_expected": matches},
                "reason": "Expected fact observed." if verified
                          else "Required evidence was missing or did not match."}
