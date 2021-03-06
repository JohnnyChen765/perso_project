?	4??K?@4??K?@!4??K?@	?ߟ?]?v??ߟ?]?v?!?ߟ?]?v?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails64??K?@??%VF?m@1?Gߤ?.{@A?1%????I?k?,	H @Y?H?5??*	?z?Gn?@2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2:?,B???!??!B?X@)S@?? k??1VؙS{V@:Preprocessing2?
NIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip::Shuffle Ș?????!?G?#?V@)Ș?????1?G?#?V@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::FiniteSkip u?(%???!??z??@)Z+??6??1L??)?? @:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??뉮??!???G????)??뉮??1???G????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismu?)?:??!ɲ?(?@)>?h??i??1?t2
?e??:Preprocessing2F
Iterator::Model?]M?????!?ϻwa@)\ ?K?b?13????<??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 35.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?ߟ?]?v?I?Aۯ'B@Q?6?#?O@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??%VF?m@??%VF?m@!??%VF?m@      ??!       "	?Gߤ?.{@?Gߤ?.{@!?Gߤ?.{@*      ??!       2	?1%?????1%????!?1%????:	?k?,	H @?k?,	H @!?k?,	H @B      ??!       J	?H?5???H?5??!?H?5??R      ??!       Z	?H?5???H?5??!?H?5??b      ??!       JGPUY?ߟ?]?v?b q?Aۯ'B@y?6?#?O@?"a
Cgradient_tape/gradient_tape/sequential_27/conv2d_55/Conv2D_8/Conv2DConv2Dt?|?R???!t?|?R???0"m
Bgradient_tape/sequential_27/conv2d_55/Conv2D_6/Conv2DBackpropInputConv2DBackpropInput????K??!?Ѓ49m??0"m
Bgradient_tape/sequential_27/conv2d_55/Conv2D_2/Conv2DBackpropInputConv2DBackpropInput[?i?o̕?! K??????0"m
Dgradient_tape/sequential_27/conv2d_55/Conv2D_5/Conv2DBackpropInput_1Conv2DBackpropInputr????Z??!??p ??"m
Bgradient_tape/sequential_27/conv2d_55/Conv2D_9/Conv2DBackpropInputConv2DBackpropInput??u??!???T?C??0"]
4sequential_28/conv2d_transpose_52/conv2d_transpose_2Conv2DBackpropInput??~???!e???e$??"m
Bgradient_tape/sequential_27/conv2d_55/Conv2D_4/Conv2DBackpropInputConv2DBackpropInput?	-?????!?O?▘??0"]
4sequential_28/conv2d_transpose_52/conv2d_transpose_3Conv2DBackpropInput"0g
SY??!?5?C???"m
Bgradient_tape/sequential_27/conv2d_55/Conv2D_7/Conv2DBackpropInputConv2DBackpropInputc<???!?i??e??0"m
Bgradient_tape/sequential_27/conv2d_55/Conv2D_8/Conv2DBackpropInputConv2DBackpropInput?CA5????!???T???0Q      Y@YIa???@a???"X@qa?W L?R@y?(t?N?N?"?

both?Your program is POTENTIALLY input-bound because 35.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?75.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 