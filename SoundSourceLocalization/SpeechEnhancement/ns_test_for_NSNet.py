#!/usr/bin/env python3
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from ns_enhance_onnx import NSnet2Enhancer

"""
    Inference script for NSnet2 baseline.
"""


def main(args):
    # check input path
    inPath = Path(args.input).resolve()
    assert inPath.exists()
    
    # Create the enhancer
    enhancer = NSnet2Enhancer(modelfile=args.model)
    # get modelname
    modelname = Path(args.model).stem
    
    if inPath.is_file() and inPath.suffix == '.wav':
        # input is single .wav file
        sigIn, fs = sf.read(str(inPath))
        if len(sigIn.shape) > 1:
            sigIn = sigIn[:, 0]
        
        outSig = enhancer(sigIn, fs)
        
        outname = './{:s}_{:s}.wav'.format(inPath.stem, modelname)
        if args.output:
            # write in given dir
            outdir = Path(args.output)
            outdir.mkdir(exist_ok=True)
            outpath = outdir.joinpath(outname)
        else:
            # write in current work dir
            outpath = Path(outname)
        
        print('Writing output to:', str(outpath))
        sf.write(outpath.resolve(), outSig, fs)
    
    elif inPath.is_dir():
        # input is directory
        if args.output:
            # full provided path
            outdir = Path(args.output).resolve()
        else:
            outdir = inPath.parent.joinpath(modelname).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        print('Writing output to:', str(outdir))
        
        fpaths = list(inPath.glob('**/*.wav'))
        for ii, path in enumerate(fpaths):
            print(f"Processing file [{ii + 1}/{len(fpaths)}]")
            sigIn, fs = sf.read(path)
            if len(sigIn.shape) > 1:
                sigIn = sigIn[:, 0]
            
            outSig = enhancer(sigIn, fs)
            # outpath = outdir / path.name
            outpath = Path(str(path).replace('采集数据', '采集数据' + '\\' + outdir.name)).resolve()
            outpath.parent.mkdir(parents=True, exist_ok=True)
            sf.write(outpath, outSig, fs)
    
    else:
        raise ValueError("Invalid input path.")


class argparse():
    pass


if __name__ == "__main__":
    args = argparse()
    args.model = 'ns_nsnet2-20ms-baseline.onnx'
    # parser.add_argument("-m", "--model", type=str, help="Path to ONNX model.", default='ns_nsnet2-20ms-baseline.onnx')
    args.input = 'G:\SmartWalker\采集数据\\16khz'
    
    main(args)
