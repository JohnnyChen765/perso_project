?	4K??zJ@4K??zJ@!4K??zJ@	8? ?#??8? ?#??!8? ?#??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails64K??zJ@Y??9??H@1G仔?d??A?n?1??I
?5??@YE?J?E??*	cX9??S@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???D???!x??V1'<@)??F???1H_??07@:Preprocessing2F
Iterator::Model??V'g??!??z{YD@)??p?Ws??1??},L,4@:Preprocessing2U
Iterator::Model::ParallelMapV2dZ???Z??!tTx?f4@)dZ???Z??1tTx?f4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate1?t?????!4?9%?E7@)n½2oՅ?1%?:?*@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??+,???!E?]???#@)??+,???1E?]???#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?h?^??!P????M@)5D?ov?1?
????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor3???/p?!??????@)3???/p?1??????@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapaũ??,??!}'u?8@)??Os?"S?1?T7}~w??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no98? ?#??I??<?̏X@Q??Ϡ6???Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Y??9??H@Y??9??H@!Y??9??H@      ??!       "	G仔?d??G仔?d??!G仔?d??*      ??!       2	?n?1???n?1??!?n?1??:	
?5??@
?5??@!
?5??@B      ??!       J	E?J?E??E?J?E??!E?J?E??R      ??!       Z	E?J?E??E?J?E??!E?J?E??b      ??!       JGPUY8? ?#??b q??<?̏X@y??Ϡ6????"I
+gradient_tape/sequential_10/dense_31/MatMulMatMulrI?A???!rI?A???0";
sequential_10/dense_31/MatMulMatMulrI?A???!rI?A???0"I
+gradient_tape/sequential_10/dense_30/MatMulMatMul???ʅ???!?V??0%??0";
sequential_10/dense_30/MatMulMatMul???ʅ???!?S?Ч?0"I
-gradient_tape/sequential_10/dense_31/MatMul_1MatMul,??-\i??!4,?)???"I
-gradient_tape/sequential_10/dense_32/MatMul_1MatMul???%7{?!Hw?)??";
sequential_10/dense_32/MatMulMatMul???%7{?!v?6;yܱ?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch?Ť??v?!r)??XG??"Y
8gradient_tape/sequential_10/dense_32/BiasAdd/BiasAddGradBiasAddGrad?Ť??v?!nz??7???"1
Nadam/Nadam/update_9/mulMul?|??%r?!<r0?Ե?Q      Y@Y?? _??@af?*??W@q?-???W@y?S??qY??"?
both?Your program is POTENTIALLY input-bound because 93.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?92.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 