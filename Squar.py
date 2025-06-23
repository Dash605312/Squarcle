import pygame as pg
import neat
import os
import math
import random

pg.init()
screen = pg.display.set_mode((1280, 720))
clock = pg.time.Clock()
font = pg.font.SysFont("Consolas", 20)

squar_xy = pg.Rect(600, 300, 50, 50)
circl_rad = 20
b_str = 1.0

def get_inputs(circl_pos, circl_spd, squar_pos):
    # Inputs: rel_x, rel_y, vel_x, vel_y
    dx = squar_pos.centerx - circl_pos.x
    dy = squar_pos.centery - circl_pos.y
    return [dx / 640, dy / 360, circl_spd.x / 10, circl_spd.y / 10]

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0

        circl_xy = pg.Vector2(random.randint(100, 1180), random.randint(100, 620))
        circl_spd = pg.Vector2()

        run = True
        time_alive = 0
        while run and time_alive < 600:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    quit()

            input_data = get_inputs(circl_xy, circl_spd, squar_xy)
            output = net.activate(input_data)
            acc = pg.Vector2(output[0] * 2 - 1, output[1] * 2 - 1) * 0.3
            circl_spd *= 0.99
            circl_spd += acc

            circl_xy += circl_spd

            # Bounce off edges
            if circl_xy.x < circl_rad:
                circl_xy.x = circl_rad
                circl_spd.x *= -b_str
            elif circl_xy.x > screen.get_width() - circl_rad:
                circl_xy.x = screen.get_width() - circl_rad
                circl_spd.x *= -b_str
            if circl_xy.y < circl_rad:
                circl_xy.y = circl_rad
                circl_spd.y *= -b_str
            elif circl_xy.y > screen.get_height() - circl_rad:
                circl_xy.y = screen.get_height() - circl_rad
                circl_spd.y *= -b_str

            # Fitness: the closer to the square, the better
            dist = math.hypot(circl_xy.x - squar_xy.centerx, circl_xy.y - squar_xy.centery)
            genome.fitness += max(0, 1.0 - dist / 800)

            # End if overlap
            if squar_xy.collidepoint(circl_xy.x, circl_xy.y):
                genome.fitness += 100
                break

            time_alive += 1

def run(config_file):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 78000)
    print("Winner Genome:", winner)
    with open("best_genome.pkl", "wb") as f:
        import pickle
        pickle.dump(winner, f)
    print("Winner genome saved to best_genome.pkl")

    print("Winner Genome:", winner)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
    print("Winner Genome:", "Done training")
