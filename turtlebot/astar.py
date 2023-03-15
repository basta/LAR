import numpy as np

import numpy as np

class Node:
    def __init__(self, x, y, dist, parent, v_map):
        self.x = x
        self.y = y
        self.dist = dist
        self.parent = parent
        self.v_map = v_map

    def get_neighbors(self) -> list["Node"]:
        ret = []
        for offset in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
            try:
                if self.v_map[self.x + offset[0]][self.y + offset[1]]:
                    continue
                ret.append(
                    Node(self.x + offset[0], self.y + offset[1], self.dist + (np.linalg.norm(np.array(offset))), self,
                         self.v_map))
                self.v_map[self.x + offset[0]][self.y + offset[1]] = 1

            except IndexError:
                continue

        return ret

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __str__(self):
        return f"Node:x={self.x},y={self.y},dist={self.dist}"

    def __repr__(self):
        return self.__str__()


def find_path(start_point: np.array, end_point: np.array, blockers: list[tuple[np.array, float]],
              size=(2,2), step=0.1):
    coord_map = [[np.array([x, y]) for y in np.arange(*([-size[1], size[1]] + [step]))] for x in
                 np.arange(*([-size[0], size[1]] + [step]))]
    visited_map = [[0 for _ in row] for row in coord_map]
    visited_map[start_point[0]][start_point[1]] = 1

    queue = set()
    start = Node(start_point[0], start_point[1], 0, None, visited_map)
    end = Node(end_point[0], end_point[1], 100000, None, visited_map)
    queue.add(start)

    def h(node):
        return ((node.x - end.x) ** 2 + (node.y - end.y) ** 2) ** 0.5

    for x, row in enumerate(coord_map):
        for y, value in enumerate(row):
            point = value
            for avoid_point, r in blockers:
                if abs(avoid_point[0] - point[0]) < r and abs(avoid_point[1] - point[1]) < r:
                    visited_map[x][y] = 2
                    break

    while queue:
        node = min(queue, key=lambda x: h(x) + x.dist)
        [queue.add(neighb) for neighb in node.get_neighbors()]
        queue.remove(node)
        if any([node for node in queue if node.x == end.x and node.y == end.y]):
            break

    for node in queue:
        if hash(node) == hash(end):
            final_end = node

    # print(final_end)
    path = []
    node = final_end
    while node:
        path.append(node)
        if node.parent:
            node = node.parent
        else:
            break

    # print(path)
    coord_path = [coord_map[point.x][point.y] for point in path[::-1]]

    def error(pixel_poses):
        err = 0
        steps = len(pixel_poses)
        direction = (pixel_poses[-1] - pixel_poses[0]) / steps

        for i in range(steps):
            err += np.linalg.norm(pixel_poses[i] - (pixel_poses[0] + direction*i))
        return err/steps

    def reduced_path(points, lookahead=200, max_err=0.2):
        last_idx = 0
        reduced = [points[0]]
        while last_idx < len(points) -2:
            errors = np.array([])
            for i in range(1, lookahead):
                if (last_idx + i) >= len(points):
                    break
                errors = np.append(errors, (error(points[last_idx:last_idx+i])))

            li = list(errors < max_err)
            last_valid_idx  = len(li) - 1 - li[::-1].index(True)
            # print(f"{last_idx=} {last_valid_idx=}")
            last_idx += last_valid_idx
            reduced.append(points[last_idx])

        return reduced

    return reduced_path(coord_path, max_err=0.1)
