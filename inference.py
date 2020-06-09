#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore, IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### Initialize any class variables desired ###
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request = None
        
        

    def load_model(self, model, device, num_req, input_size, output_size, cpu_ext=None):
        ### Load the model ###
        
        log.info("Loading the model...")
        model_bin = os.path.splitext(model)[0] + ".bin"
        self.net = IENetwork(model=model, weights=model_bin)
        
        ### Check for supported layers ###
        log.info("Checking the supported layers...")
        
       
        log.info("Initailze plugin for the device %s " % device)
        self.plugin = IEPlugin(device=device)
            
        if cpu_ext and device == 'CPU':
            self.plugin.add_cpu_extension(cpu_ext)
        
        if self.plugin.device == 'CPU':
            supp_layers = self.plugin.get_supported_layers(self.net)
            not_supp_layers = [l for l in self.net.layers.keys() if l not in supp_layers]
            
            if len(not_supp_layers) != 0:
                log.error("The following layers are not supported for your device %s :\n %s" % (self.plugin.device, ', '.join(not_supp_layers)))
                sys.exit(1)

        ### Add any necessary extensions ###
        
        if num_req == 0:
            self.net_plugin = self.plugin.load(network = self.net)
        else:
            self.net_plugin = self.plugin.load(network = self.net, num_requests = num_req)
        
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        
        assert len(self.net.inputs.keys()) == input_size, "Input size should be of %s !" % len(self.net.inputs)
        
        assert len(self.net.outputs) == output_size, "Output size should be of %s !" % len(self.net.outputs)
        ### Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        ### Return the shape of the input layer ###
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, req_id, frame):
        ### Start an asynchronous request ###
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.infer_request = self.net_plugin.start_async(request_id = req_id, inputs = {self.input_blob: frame})
        
        return self.net_plugin

    def wait(self, req_id):
        ### Wait for the request to be complete. ###
        ### Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        wait_process = self.net_plugin.requests[req_id].wait(-1)
        return wait_process

    def get_output(self, req_id, output = None):
        ### Extract and return the output results
        ### Note: You may need to update the function parameters. ###
       
        if output:
            res = self.infer_request.outputs[output]
        else:
            res = self.net_plugin.requests[req_id].outputs[self.out_blob]
            
        return res
