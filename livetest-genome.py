import pygame as pg
import neat
import os
import math

pg.init()
screen = pg.display.set_mode((1280, 720))
clock = pg.time.Clock()
font = pg.font.SysFont("Consolas", 20)

squar_xy = pg.Rect(600, 300, 50, 50)
circl_rad = 20
b_str = 1.0

def get_inputs(circl_pos, circl_spd, squar_pos):
    dx = squar_pos.centerx - circl_pos.x
    dy = squar_pos.centery - circl_pos.y
    return [dx / 640, dy / 360, circl_spd.x / 10, circl_spd.y / 10]

def run_best(config_path, genome_path):
    # Load config & genome
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    with open(genome_path, "rb") as f:
        import pickle
        genome = pickle.load(f)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    circl_xy = pg.Vector2(100, 100)
    circl_spd = pg.Vector2()

    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        inputs = get_inputs(circl_xy, circl_spd, squar_xy)
        output = net.activate(inputs)
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

        screen.fill("green")
        pg.draw.rect(screen, "blue", squar_xy)
        pg.draw.circle(screen, "red", circl_xy, circl_rad)
        pg.display.flip()
        clock.tick(60)

    pg.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    genome_path = os.path.join(local_dir, "best_genome.pkl")  # Save your best genome here after training
    run_best(config_path, genome_path)
