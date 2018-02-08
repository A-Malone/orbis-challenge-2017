from PythonClientAPI.Game.Enums import TileType, Direction, Team
from PythonClientAPI.Game.PointUtils import mod_point, mod_taxi_cab_distance
from PythonClientAPI.DataStructures.Collections import Queue, PriorityQueue, recursively_flatten_list
from PythonClientAPI.Game import PointUtils
from PythonClientAPI.Game.Entities import FriendlyUnit, EnemyUnit, Tile
from PythonClientAPI.Game.Enums import Direction, MoveType, MoveResult
from PythonClientAPI.Game.World import World

import heapq

def add2(a, b, dim):
    return PointUtils.mod_point(PointUtils.add_points(a,b), dim)

COMPLETION_BENEFIT = 0.1
ENEMY_STRENGTH_FACTOR = 0.01

def get_potential_map(world, nest_completion_set, friendly_units, enemy_units):
    """ Builds a potential map to help guide local decision making. Fireflies
    will occaisionally deviate from their original assignment to move down 
    the potential gradient, guiding them towards enemy pushes and filling 
    in nests."""
    potential_map = {}

    dim = (world.get_width(), world.get_height())

    for p in nest_completion_set:
        potential_map[p] = potential_map.get(p, 1) - COMPLETION_BENEFIT

    for e in enemy_units:
        potential_map[e.position] = potential_map.get(e.position, 1) - ENEMY_STRENGTH_FACTOR * e.health**2
        
        for d in ((1,0), (-1,0), (0,1), (0,-1)):
            p = e.position
            for r in range(1,2):
                p = add2(p,d , dim)
                if(world.is_wall(p)):
                    break
                potential_map[p] = potential_map.get(p, 1) - (ENEMY_STRENGTH_FACTOR * e.health**2) / r
    
    return potential_map

# Cost-based breadth-first search
def get_cheapest_point_from(world, cost_map, source, condition):
    """Adapted BFS to find the cheapest point using a potential map, however 
    since the cost function is not necessarily positive for every step, it's
    possible to miss the optimal point."""
    queue = PriorityQueue()
    visited = set()
    movement_costs = {}

    queue.add(source, 0)
    visited.add(source)
    movement_costs[source] = 0

    while not (queue.is_empty()):
        cursor = queue.poll()
        neighbours = world.get_neighbours(cursor)

        for direction in Direction.ORDERED_DIRECTIONS:
            neighbour = neighbours[direction]

            cost = movement_costs[cursor] + cost_map.get(neighbour, 1)
            if not ((neighbour in visited) or world.is_wall(neighbour)):
                movement_costs[neighbour] = cost
                queue.add(neighbour, cost)
                visited.add(neighbour)

        if condition(cursor): return cursor

    return None

def get_cheapest_capturable_tile_from(world, cost_map, point, excluding_points):
    if not world.api._position_to_tile_cache: world.api._create_position_to_tile_cache()
    target = get_cheapest_point_from(world, cost_map, point, lambda p: (p in world.api._position_to_tile_cache) and (not world.api._position_to_tile_cache[p].is_friendly()) and (not world.api._position_to_tile_cache[p].is_permanently_owned()) and ((not excluding_points) or (p not in excluding_points)))
    if target: return world.api._position_to_tile_cache[target]
    return None

# A* path-finding
def get_cheapest_path(world, cost_map, start, end, avoid):
    """Adapted A* to find the cheapest path to a given point using a potential map."""

    if start == end: return [end]
    if world.is_wall(start) or world.is_wall(end): return None

    queue = PriorityQueue()

    queue.add(start, 0)

    inverted_tree = {}
    movement_costs = {}

    inverted_tree[start] = None
    movement_costs[start] = 0

    while not queue.is_empty():
        current = queue.poll()

        neighbours = world.get_neighbours(current)
        for direction in Direction.ORDERED_DIRECTIONS:
            neighbour = neighbours[direction]
            if world.is_wall(neighbour) or (avoid and (neighbour in avoid)):
                continue
            cost = movement_costs[current] + cost_map.get(neighbour, 1)
            if (neighbour not in movement_costs) or (cost < movement_costs[neighbour]):
                movement_costs[neighbour] = cost
                queue.add(neighbour,
                            cost + mod_taxi_cab_distance(neighbour, end, world.get_width(), world.get_height()))
                inverted_tree[neighbour] = current

        if current == end:
            path = []
            cursor = end
            peek_cursor = inverted_tree[cursor]
            while peek_cursor:
                path.append(cursor)
                cursor = peek_cursor
                peek_cursor = inverted_tree[cursor]
            path.reverse()
            return path

    return None