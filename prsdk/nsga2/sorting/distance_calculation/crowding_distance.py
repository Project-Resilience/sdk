"""
Implementation of distance calculation that uses crowding distance from NSGA-II
"""
from prsdk.nsga2.candidate.candidate import Candidate
from prsdk.nsga2.sorting.distance_calculation.distance_calculator import DistanceCalculator


class CrowdingDistanceCalculator(DistanceCalculator):
    """
    Calculates NSGA-II crowding distance
    """
    def calculate_distance(self, front: list[Candidate]) -> None:
        """
        Calculate crowding distance of each candidate in front and set it as the distance attribute.
        Candidates are assumed to already have metrics computed.
        """
        for c in front:
            c.sorting_metrics["distance"] = 0

        # Front is sorted by each metric
        for m in front[0].metrics.keys():
            front.sort(key=lambda c: c.metrics[m])
            # Standard NSGA-II Crowding Distance calculation
            obj_min = front[0].metrics[m]
            obj_max = front[-1].metrics[m]
            front[0].sorting_metrics["distance"] = float('inf')
            front[-1].sorting_metrics["distance"] = float('inf')
            if obj_max != obj_min:
                for i in range(1, len(front) - 1):
                    distance = (front[i+1].metrics[m] - front[i-1].metrics[m]) / (obj_max - obj_min)
                    front[i].sorting_metrics["distance"] += distance
