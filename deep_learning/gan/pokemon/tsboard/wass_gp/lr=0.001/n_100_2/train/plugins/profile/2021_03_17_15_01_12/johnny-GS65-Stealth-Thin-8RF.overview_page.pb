?	Ǻ???z?@Ǻ???z?@!Ǻ???z?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-Ǻ???z?@6?!??"@1$???9?@A??_YiR??I??I`;!@*	?Q??8?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2B?v????!?&@?+X@)?R\U????1?!3αV@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip ?u?????!?属?@)R%?S;??1?$??m@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle ?-?????!?/?Ь?@)?-?????1?/?Ь?@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch}!??????!???p???)}!??????1???p???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??-?R??!V?k?7F@)WBwI???1???R???:Preprocessing2F
Iterator::ModelC?ʠ????!?'?7??
@)Ƣ??dpd?1<{?#???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI@???-{@Q^??&$X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	6?!??"@6?!??"@!6?!??"@      ??!       "	$???9?@$???9?@!$???9?@*      ??!       2	??_YiR????_YiR??!??_YiR??:	??I`;!@??I`;!@!??I`;!@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q@???-{@y^??&$X@?"o
Cgradient_tape/sequential_33/conv2d_65/Conv2D_3/Conv2DBackpropFilterConv2DBackpropFilterC1ԧ??!C1ԧ??0"]
4sequential_34/conv2d_transpose_64/conv2d_transpose_3Conv2DBackpropInput\??f?G??!???K????"m
Bgradient_tape/sequential_33/conv2d_67/Conv2D_8/Conv2DBackpropInputConv2DBackpropInputH???U??!/'3?T???0"m
Dgradient_tape/sequential_33/conv2d_67/Conv2D_2/Conv2DBackpropInput_1Conv2DBackpropInputVwåa4??!??l޷?"m
Dgradient_tape/sequential_33/conv2d_67/Conv2D_8/Conv2DBackpropInput_1Conv2DBackpropInputX-K??1??!Z?v??*??"m
Bgradient_tape/sequential_33/conv2d_67/Conv2D_4/Conv2DBackpropInputConv2DBackpropInput5??????!??y??2??0"k
@gradient_tape/sequential_33/conv2d_67/Conv2D/Conv2DBackpropInputConv2DBackpropInput??k??ؔ?!`J?????0"m
Bgradient_tape/sequential_33/conv2d_67/Conv2D_6/Conv2DBackpropInputConv2DBackpropInputhl3n?Д?!???M?g??0"m
Bgradient_tape/sequential_33/conv2d_67/Conv2D_2/Conv2DBackpropInputConv2DBackpropInput4q?t???!???????0"m
Bgradient_tape/sequential_33/conv2d_67/Conv2D_3/Conv2DBackpropInputConv2DBackpropInput?ZAI??!q?w???0Q      Y@YIa???@a???"X@q6R?L?A"@y??E?[J?"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 