?	5?? %?@5?? %?@!5?? %?@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-5?? %?@????Gjn@1?*?b?z@A?C9Ѯ??I??1=aI@*	?Q??r?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV20e???.??!??-aD#X@)	?Į????1??c?V@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip j?!?
??!>???@)?ᱟ?R??1?UndG)@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle FCƣT?!?????@)FCƣT?1?????@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??Ũk???!MGq?K)??)??Ũk???1MGq?K)??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Md????!|???@)?$?@??1???????:Preprocessing2F
Iterator::ModelmY?.???!??O?s?@))_?BFg?1f??V?;??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 36.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*noI?Xz:ӊB@Q=???,uO@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????Gjn@????Gjn@!????Gjn@      ??!       "	?*?b?z@?*?b?z@!?*?b?z@*      ??!       2	?C9Ѯ???C9Ѯ??!?C9Ѯ??:	??1=aI@??1=aI@!??1=aI@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?Xz:ӊB@y=???,uO@?	"k
@gradient_tape/sequential_29/conv2d_59/Conv2D/Conv2DBackpropInputConv2DBackpropInputýf?^??!ýf?^??0"m
Bgradient_tape/sequential_29/conv2d_59/Conv2D_7/Conv2DBackpropInputConv2DBackpropInput???"?-??!?]?S???0"m
Dgradient_tape/sequential_29/conv2d_59/Conv2D_8/Conv2DBackpropInput_1Conv2DBackpropInput?ݨ߫??!??2????"m
Bgradient_tape/sequential_29/conv2d_59/Conv2D_2/Conv2DBackpropInputConv2DBackpropInput??I??!??3????0"m
Bgradient_tape/sequential_29/conv2d_59/Conv2D_5/Conv2DBackpropInputConv2DBackpropInput????ԓ?!/õ???0"m
Bgradient_tape/sequential_29/conv2d_59/Conv2D_1/Conv2DBackpropInputConv2DBackpropInput?d?H{??!(???߿?0"m
Dgradient_tape/sequential_29/conv2d_59/Conv2D_5/Conv2DBackpropInput_1Conv2DBackpropInput?˸??5??!?|^??V??"m
Bgradient_tape/sequential_29/conv2d_59/Conv2D_6/Conv2DBackpropInputConv2DBackpropInputo?ߖ4??!m:?3???0"m
Bgradient_tape/sequential_29/conv2d_59/Conv2D_4/Conv2DBackpropInputConv2DBackpropInput??h
??![0S????0"m
Bgradient_tape/sequential_29/conv2d_59/Conv2D_3/Conv2DBackpropInputConv2DBackpropInput?m=W???!??}??0Q      Y@YIa???@a???"X@q?#?F??S@y?3?U!?O?"?

both?Your program is POTENTIALLY input-bound because 36.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?79.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 