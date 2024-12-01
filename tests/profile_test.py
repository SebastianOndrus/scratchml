import cProfile
import pstats
import unittest
import io

def run_tests_with_profiling():
    profiler = cProfile.Profile()
    profiler.enable()

    suite = unittest.TestLoader().loadTestsFromName('tests.models.test_random_forest.Test_Random_Forest')
    unittest.TextTestRunner().run(suite)

    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
    ps.print_stats()
    
    # Filter the output
    output = s.getvalue().split('\n')
    for line in output:
        if 'scratchml' in line:  # Filter lines containing 'scratchml'
            print(line)

if __name__ == "__main__":
    run_tests_with_profiling()