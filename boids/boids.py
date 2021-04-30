# boids.py
#
# Bryan Daniels
# 2021/4/40
#
# Helper code for launching boids javascript simulation from a jupyter notebook.
#

def setupBoidsSimulation(attractiveFactor,alignmentFactor,avoidFactor,
    visualRange=75,numBoids=100,drawTrail=False,
    originalFilename='./boids/index.html',
    modifiedFilename='./boids/index-modified.html'):
    """
    Write a modified boids simulation HTML file with parameters set
    according to input arguments.
    
    Units of the force factor parameters are rescaled such that 1 corresponds
    to the default value in the original simulation.
    """
    
    # read original HTML simulation file
    fin = open(originalFilename,'r')
    originalHTML = fin.read()
    fin.close()
    
    # change units of parameters such that 1 corresponds to the default value
    attractiveFactorDefault = 0.005
    alignmentFactorDefault = 0.05
    avoidFactorDefault = 0.05
    attractiveFactor = attractiveFactor * attractiveFactorDefault
    alignmentFactor = alignmentFactor * alignmentFactorDefault
    avoidFactor = avoidFactor * avoidFactorDefault
    
    # replace
    modifiedHTML = originalHTML.replace(
        'const attractiveFactor = 0.005;',
        'const attractiveFactor = {};'.format(attractiveFactor)).replace(
        'const alignmentFactor = 0.05;',
        'const alignmentFactor = {};'.format(alignmentFactor)).replace(
        'const avoidFactor = 0.05;',
        'const avoidFactor = {};'.format(avoidFactor)).replace(
        'const visualRange = 75;',
        'const visualRange = {};'.format(visualRange)).replace(
        'const numBoids = 100;',
        'const numBoids = {};'.format(numBoids)).replace(
        'const DRAW_TRAIL = false;',
        'const DRAW_TRAIL = {};'.format(str(drawTrail).lower()))

    fout = open(modifiedFilename,'w')
    fout.write(modifiedHTML)
    fout.close()
    
