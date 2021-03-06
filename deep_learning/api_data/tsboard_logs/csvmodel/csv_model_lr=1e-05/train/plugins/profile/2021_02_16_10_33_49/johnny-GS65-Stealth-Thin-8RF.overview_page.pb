?	?KToL@?KToL@!?KToL@	U??,?_??U??,?_??!U??,?_??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?KToL@??SDJ@1??P?????A)=?K?e??I??[???@YϞ??$x??*	?K7?A0S@2U
Iterator::Model::ParallelMapV2????je??!?(A?g7@)????je??1?(A?g7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat~?*O ???!LB???:@)?@???1?]?O?5@:Preprocessing2F
Iterator::Modelf??E???!?EKEiF@)???Ր?1mbU ?j5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate)&o?????!?3R?M?5@)?!??gx??1???h??(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceTUh ??|?!??A?R"@)TUh ??|?1??A?R"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????!n?????K@)??Gߤip?1L\?]??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor;?f??o?!贐K?@);?f??o?1贐K?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????W??!Z??>V7@)??;??~V?1/j&???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9U??,?_??I?.???X@Q??V_ɬ??Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??SDJ@??SDJ@!??SDJ@      ??!       "	??P???????P?????!??P?????*      ??!       2	)=?K?e??)=?K?e??!)=?K?e??:	??[???@??[???@!??[???@B      ??!       J	Ϟ??$x??Ϟ??$x??!Ϟ??$x??R      ??!       Z	Ϟ??$x??Ϟ??$x??!Ϟ??$x??b      ??!       JGPUYU??,?_??b q?.???X@y??V_ɬ???";
sequential_13/dense_39/MatMulMatMul??en???!??en???0"I
+gradient_tape/sequential_13/dense_40/MatMulMatMul6q?gޱ??!]??>&???0"I
+gradient_tape/sequential_13/dense_39/MatMulMatMul?d???s??!??m? ???0";
sequential_13/dense_40/MatMulMatMul`8?F's??!??(?ʒ??0"I
-gradient_tape/sequential_13/dense_40/MatMul_1MatMul?+??4??!??)?
???"Y
8gradient_tape/sequential_13/dense_41/BiasAdd/BiasAddGradBiasAddGrad?????z?!=???A???"I
-gradient_tape/sequential_13/dense_41/MatMul_1MatMul?????z?!I??;???";
sequential_13/dense_41/MatMulMatMul?N???z?!꥕FE]??0"1
Nadam/Nadam/update/mul_2Mul`8?F'sv?!pi?wĴ?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch`8?F'sv?!?,s/?+??Q      Y@Yu?E]t@a颋.??W@q%EX`??T@yxF????"?
both?Your program is POTENTIALLY input-bound because 93.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?4.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?82.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 