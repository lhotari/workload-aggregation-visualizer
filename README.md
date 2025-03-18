# Multi-Tenant Workload Aggregation: Interactive Simulation & Visualization Tool

## Overview

This simulation and visualization is inspired by Andrew Warfield's talk at FAST '23: ["Building and Operating a Pretty Big Storage System (My Adventures in Amazon S3)"](https://www.youtube.com/watch?v=sc3J4McebHE&t=1333s).

This simulation was created to provide visualization of workload aggregation for a guest lecture in the Aalto University course "Networking at Scale and Advanced Applications" on March 18, 2025, covering [Autoscaling challenges of multitenant SaaS platforms](https://lhotari.github.io/workload-aggregation-visualizer/presentation/Autoscaling%20challenges%20of%20multitenant%20SaaS%20platforms.pdf). You can find the slides [here](https://lhotari.github.io/workload-aggregation-visualizer/presentation/Autoscaling%20challenges%20of%20multitenant%20SaaS%20platforms.pdf).

* Multi-tenancy significantly reduces the peak-to-average ratio of overall system load
  * In general, systems must provision for peak demand, not average load to meet performance quality requirements
  * Lower peak-to-average ratios directly translate to fewer overprovisioned resources
  * This results in improved resource utilization and cost efficiency

Please note that this script contains many assumptions about the simulated workloads and doesn't necessarily match reality. The goal is to visualize the effects of workload aggregation on system load. The [presentation](https://lhotari.github.io/workload-aggregation-visualizer/presentation/Autoscaling%20challenges%20of%20multitenant%20SaaS%20platforms.pdf) contains more details about cases where workloads are correlated and don't follow the model used in this simulation.

## Features of the simulation

- **Interactive Visualization:** Dynamically adjust the number of aggregated workloads (k × k grid)
  - **Keyboard Controls:**
    - Use arrow keys (↑/↓/←/→) to adjust the number of workloads
    - Toggle between visualization modes with `Tab` key
    - Save screenshots with `s` key
    - Regenerate workloads by pressing `Space` or `Enter` key
    - Press `q` or `Esc` key to exit
- **Two Visualization Modes:**
  - **Workload Mode:** Shows individual workloads and their aggregation
  - **Overprovisioning Mode:** Visualizes how the required overprovisioning factor decreases as more workloads are aggregated
- **Batch mode:**
  - Generate image files and exit

## Installation

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/

2. Run the script directly (dependencies will be automatically installed when using uv):

   ```shell
   ./workload-aggregation-visualizer.py
   ```

## Usage

Run the script to start in UI mode, the console will contain information about keybindings:

```shell
./workload-aggregation-visualizer.py
```

pass `--help` to see complete set of command line options

## Visualization example

Images were generated with this command:

```shell
./workload-aggregation-visualizer.py --output-dir examples --format svg --plot-overprovisioning --batch
```
_(Some images are omitted to simplify the example.)_
![workload 1x1 - single workload visualization](examples/aggregated_workloads_1x1.svg)
![workload 5x5 - aggregation of 25 workloads](examples/aggregated_workloads_5x5.svg)
![workload 7x7 - aggregation of 49 workloads](examples/aggregated_workloads_7x7.svg)
![workload 10x10 - aggregation of 100 workloads](examples/aggregated_workloads_10x10.svg)
![Graph showing how overprovisioning factor decreases as k increases from 1 to 10](examples/overprovisioning_factor_k1-10.svg)

## Presentation that includes this visualization

* Guest lecture for Aalto University course “Networking at Scale and Advanced Applications” on March 18, 2025 about [Autoscaling challenges of multitenant SaaS platforms](https://lhotari.github.io/workload-aggregation-visualizer/presentation/Autoscaling%20challenges%20of%20multitenant%20SaaS%20platforms.pdf) by Lari Hotari

## Resources related to multi-tenant SaaS and economics of scale in cloud services

These aren't necessarily directly related to the visualization in this project. These resources were the inspiration to create this visualization as well as some of the resources for the [related presentation](https://lhotari.github.io/workload-aggregation-visualizer/presentation/Autoscaling%20challenges%20of%20multitenant%20SaaS%20platforms.pdf).

* [Andy Warfield: Building and Operating a Pretty Big Storage System (My Adventures in Amazon S3), part "Individual workloads are bursty", presentation at FAST'23](https://www.youtube.com/watch?v=sc3J4McebHE&t=1333s)
* [Marc Brooker: Surprising Scalability of Multitenancy, blog post 2023-03-23](https://brooker.co.za/blog/2023/03/23/economics.html)
* [Marc Brooker: Surprising Economics of Load-Balanced Systems, blog post 2020-08-06](https://brooker.co.za/blog/2020/08/06/erlang.html)
* [Jack Vanlightly: On the future of cloud services and BYOC, blog post 2023-09-25](https://jack-vanlightly.com/blog/2023/9/25/on-the-future-of-cloud-services-and-byoc)
* Elhemali\, M\.\, Gallagher\, N\.\, Gordon\, N\.\, Idziorek\, J\.\, Krog\, R\.\, Lazier\, C\.\, Mo\, E\.\, Mritunjai\, A\.\, Perianayagam\, S\.\, Rath\, T\.\, Sivasubramanian\, S\.\, Sorenson III\, J\.C\.\, Sosothikul\, S\.\, Terry\, D\.\, & Vig\, A\. \(2022\)\. __Amazon DynamoDB: A scalable\, predictably performant\, and fully managed NoSQL database service\.__ Retrieved from _https://www\.amazon\.science/publications/amazon\-dynamodb\-a\-scalable\-predictably\-performant\-and\-fully\-managed\-nosql\-database\-service_
  * _[Conference presentation on YouTube](https://www.youtube.com/watch?v=9AkgiEJ_dA4)_
* Ed Huang’s blog post series about TiDB's resource control framework
  * _[The Road To Serverless: Intro & Why](https://me.0xffff.me/dbaas1.html)_
  * _[The Road To Serverless: Storage Engine](https://me.0xffff.me/dbaas2.html)_
  * _[The Road To Serverless: Multi\-tenanc](https://me.0xffff.me/dbaas3.html)_  _[y](https://me.0xffff.me/dbaas3.html)_
* Jack Vanlightly’s blog post series
  * _[The Architecture of Serverless Data Systems](https://jack-vanlightly.com/blog/2023/11/14/the-architecture-of-serverless-data-systems)_
* Povzner\, A\.\, Mahajan\, P\.\, Gustafson\, J\.\, Rao\, J\.\, Juma\, I\.\, Min\, F\.\, Sridharan\, S\.\, Bhatia\, N\.\, Attaluri\, G\.\, Chandra\, A\.\, Kozlovski\, S\.\, Sivaram\, R\.\, Bradstreet\, L\.\, Barrett\, B\.\, Shah\, D\.\, Jacot\, D\.\, Arthur\, D\.\, Dagostino\, R\.\, McCabe\, C\.\, Obili\, M\. R\.\, Prakasam\, K\.\, Sancio\, J\. G\.\, Singh\, V\.\, Nikhil\, A\.\, & Gupta\, K\. \(2023\)\.  __Kora: A Cloud\-Native Event Streaming Platform for Kafka\.__  Proceedings of the VLDB Endowment\, 16\(12\)\, 3822\-3834\.  _[https://doi\.org/10\.14778/3611540\.3611567](https://doi.org/10.14778/3611540.3611567)_
* AWS blog posts by David Yanacek
  * _[Fairness in multi\-tenant systems](https://aws.amazon.com/builders-library/fairness-in-multi-tenant-systems/)_
  * _[Using load shedding to avoid overload](https://aws.amazon.com/builders-library/using-load-shedding-to-avoid-overload/)_

## License

This project is licensed under the MIT License with attribution requirements - see the [LICENSE](LICENSE) file for details.

## Author

Created by Lari Hotari