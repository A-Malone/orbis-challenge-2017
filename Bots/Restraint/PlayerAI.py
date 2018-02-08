from PythonClientAPI.Game import PointUtils
from PythonClientAPI.Game.Entities import FriendlyUnit, EnemyUnit, Tile
from PythonClientAPI.Game.Enums import Direction, MoveType, MoveResult
from PythonClientAPI.Game.World import World

from collections import deque
from enum import Enum
import random
import time

import MapUtils

def get_walls(world):
    """Returns the set of all walls"""
    return set(((x,y) for x in range(world.get_width()) for y in range(world.get_height()) if world.is_wall((x,y))))

def get_spawns(world):
    """Returns the original friendly nest position, enemy nest position, and 
    the separation between the two."""
    my_start = world.get_friendly_nest_positions()[0]
    their_start = world.get_enemy_nest_positions()[0]

    return my_start, their_start, len(world.get_shortest_path(my_start, their_start, None))

def tile_nests(world, walls):
    """ Pick nest positions using a set starting pattern and breadth first traversal of the board """
    
    # World parameters
    dim = (world.get_width(), world.get_height())
    f_spawn, e_spawn, spawn_distance = get_spawns(world)

    # ---- STARTING NESTS
    # For fast expansion, try to create a 5-nest cluster in an X-shaped pattern:
    #   X_X     This provides enough early economy to expand and rapidly create
    #   _X_     unclustered nests accross the rest of the plane.
    #   x_x
    potential_starting_nest_locations = (MapUtils.add2((x,y), f_spawn, dim) for x in [-1,1] for y in [-1,1])    
    nest_locations = {p for p in potential_starting_nest_locations if (world.is_within_bounds(p) and not world.is_wall(p))}

    nest_adjacent = set()
    for nest in nest_locations:
        nest_adjacent = nest_adjacent | {x.position for x in world.get_tiles_around(nest).values() if x.position != f_spawn}

    # ---- BREADTH FIRST NEST TILING
    # Perform a breadth first traversal of the section of the board closer to 
    # my original spawn than the enemy's to choose positions to build 
    # non_overlapping nests in.

    queue = deque()
    queue.append(f_spawn)
    visited = nest_locations.copy()

    # Gets adjacent positions
    adjacent_pos = lambda x: (x.position for x in world.get_tiles_around(x).values())

    # Checks to see if my spawn is closer to a random spot
    my_spawn_closer = lambda x: world.get_shortest_path_distance(x, f_spawn) < world.get_shortest_path_distance(x, e_spawn)

    # Breadth first traversal loop
    while (len(queue) > 0):
        pos = queue.pop()

        adjacent = set(adjacent_pos(pos)) - walls
        intersects_nest = (pos in nest_locations) or (pos in nest_adjacent) or (adjacent & nest_adjacent)

        # Check to see if it intersects another nest
        if (not intersects_nest):
            # Check to see if all non-wall sides are reachable
            if (all((world.get_shortest_path(f_spawn, p, nest_locations | {pos}) != None for p in adjacent))):
                nest_locations.add(pos)
                nest_adjacent = nest_adjacent | adjacent

        # Add adjacent tiles
        for n in adjacent:
            if ((not n in visited) and my_spawn_closer(n)):
                queue.appendleft(n)
                visited.add(n)

    return nest_locations, nest_adjacent

# Represents tasks that the AI should perform. These are created so that units
# do not need to be assigned new target positions every cycle.
class Task(object):
    def __init__(self, point, path):
        self.target = point
        self.path = path
        self.complete = False
    
    def get_next_move(self):
        self.complete = len(self.path) == 1
        return self.path.pop()

# The player AI itself
class PlayerAI:

    def __init__(self):
        """
        Any instantiation code goes here
        """
        
        # World params
        self.spawn_distance = 0

        # Nest planning
        self.done_init = False
        self.wall_set = None
        self.planned_nest_set = None
        self.nest_completion_set = None

        # Task mapping
        self.uuid_task_map = {}

        self.turn = 0

    # ---- UNIT DISPATCH
    def get_task_for_unit(self, world, unit, pos_density, neg_density):
        """ Dispatches units based on the current world state """
        if (unit.health > 3*neg_density and unit.health > 5):
            return self.get_attack_task_for_unit(world, unit)
        else:
            return self.get_expand_task_for_unit(world, unit)
    
    def get_expand_task_for_unit(self, world, unit):
        """Finds the closest capturable tile to the unit, and sends it towards it"""

        target = world.get_closest_capturable_tile_from(unit.position, self.planned_nest_set).position 

        if (target):
            path = world.get_shortest_path(unit.position, target, self.planned_nest_set)
            if (path):
                return Task(target, path[::-1])
            else:
                return self.get_attack_task_for_unit(world, unit)
        else:
            return None

    def get_attack_task_for_unit(self, world, unit):
        """Picks the largest enemy cluster and sends the unit towards it's closest nest"""
        target_clusters = world.get_enemy_nest_clusters()
        max_cluster = max(target_clusters, key=len)

        target = min(max_cluster, key=lambda x: world.get_shortest_path_distance(unit.position, x))
        path = world.get_shortest_path(unit.position, target, self.planned_nest_set)
        if (path):
            return Task(target, path[::-1])

    # ---- BASE INTERFACE
    def do_move(self, world, friendly_units, enemy_units):
        """
        This method will get called every turn.
        
        :param world: World object reflecting current game state
        :param friendly_units: list of FriendlyUnit objects
        :param enemy_units: list of EnemyUnit objects
        """
        # Fly away to freedom, daring fireflies
        # Build thou nests
        # Grow, become stronger
        # Take over the world

        start = time.clock()
        
        # ---- MAP ANALYSIS AND PLANNING
        if (not self.done_init):
            f_spawn, e_spawn, self.spawn_distance = get_spawns(world)
            self.wall_set = get_walls(world)
            self.planned_nest_set, self.nest_completion_set = tile_nests(world, self.wall_set)
            self.done_init = True

        # Objectives
        self.nest_completion_set = {x for x in self.nest_completion_set if world.get_tile_at(x).is_neutral()}
        potential_map = MapUtils.get_potential_map(world, self.nest_completion_set, friendly_units, enemy_units)

        # Update cached variables
        active_agents = {x.uuid for x in friendly_units}
        self.uuid_task_map = {k:v for k,v in self.uuid_task_map.items() if k in active_agents}
        
        # Statistics
        neg_strength = sum((x.health for x in enemy_units))
        pos_strength = sum((x.health for x in friendly_units))
        
        neg_density = neg_strength / len(enemy_units)
        pos_density = pos_strength / len(friendly_units)
        
        fraction_explored = 1  - (len(world.get_neutral_tiles()) / (world.get_width() * world.get_height()))
        
        # Lookups
        pos_enemy_lookup = world.get_position_to_enemy_dict()
        
        idle_units = list(unit for unit in friendly_units if not unit.uuid in self.uuid_task_map or self.uuid_task_map[unit.uuid].complete)

        # Print turn statistics
        print("f:{} e:{} - {}/{} idle".format(pos_strength, neg_strength, len(idle_units), len(friendly_units)))

        # ---- UNIT LOOP
        for unit in friendly_units:

            # Ensure that we don't time out
            current = time.clock()
            if (current - start > 0.55):
                break

            idle = not unit in self.uuid_task_map
            curr_task = None
            current_move = None

            # ---- GLOBAL STRATEGY
            # Units are dispatched to achieve global objectives, but may
            # divert from their course and achieve local goals along the way.
            if (idle):
                curr_task = self.get_task_for_unit(world, unit, pos_density, neg_density)
                if (curr_task):
                    self.uuid_task_map[unit.uuid] = curr_task
            else:
                curr_task = self.uuid_task_map[unit.uuid]
            
            if (curr_task):
                current_move = curr_task.get_next_move()

            # ---- LOCAL STRATEGY
            # In addition to top-down control based on game state, units can
            # make local decisions about attacking, defending and resting
            # based on enemies and a potential map. 

            neighbours = list(world.get_neighbours(unit.position).values())
            local_potential = potential_map.get(unit.position, 1)

            # Some probability of simply staying put and powering up
            if (fraction_explored > 0.9 and random.random() < 0.15):
                current_move = None
                if curr_task: curr_task.complete        # Set the current task to complete so that unit is reassigned

            # Probability based local behavior
            for n in neighbours:
                delta = local_potential - potential_map.get(n, 1)
                if (delta > 1 or (delta > 0 and random.random() < delta)):
                    current_move = n
                    if curr_task: curr_task.complete        # Set the current task to complete so that unit is reassigned

            # Priority #1 trading with enemies, then other moves
            if (not current_move in pos_enemy_lookup):
                adjacent_enemies = [x for x in neighbours if x in pos_enemy_lookup]
                if (adjacent_enemies):
                    current_move = max(adjacent_enemies, key=lambda x: pos_enemy_lookup[x].health)
                    if curr_task: curr_task.complete        # Set the current task to complete so that unit is reassigned

            # ---- PERFORM ACTION
            # Perform the chosen move itself
            if (current_move):
                try:
                    world.move(unit, current_move)
                except Exception as e:
                    print('An exception occurred: {}'.format(e))
        
        self.turn += 1