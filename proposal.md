# Title:
Low Cost Localization Using Google Street View

### Summary:
We intend to introduce a new approach to localization that provides a higher level of accuracy than a traditional GPS using cost-effective sensors. Our sensor suite will consist of a single iPhone to provide image data as well as IMU outputs. We will calculate the homography between the camera image features and Google Street View panorama features to find the transformation between the vehicle frame and the Street View camera frame. We know that the Google Street View image locations have very accuracte GPS readings compared to a traditional GPS so we will use this fact to position ourself more accurately in the environment. 

### Splash images
One or two graphics that capture and communicate the problem and proposed solution to technical but non-expert audiences.  Don't use images that aren't yours, and make sure this figure in isolation still effectively communicates your project summary above. 

### Project git repo(s):
https://github.com/alexanderswerdlow/MonocularStreetViewLocalization

## Big picture 

### What is the overall problem that this and related research is trying to solve?
We are attempting to solve the problem of more exact GPS measurement. Almost every car today has a GPS and for those that don't, the driver usually has a cell phone. While these GPS systems are good for general navigation, increasing the accuracy of these systems would allow the data to be used to create an autonomous vehicle for a fraction of the current cost. 

High costs and low scalability of autonomous vehicle implementations will lead to a slow rollout and limited accessibility of the technology to just the mapped regions. These self-driving car companies create feature-rich and dense maps of a limited region. Our methods can allow precise localization and therefore motion/route planning on a limited and cost-effective hardware suite. GPS already provide lane indications in navigation (i.e. Google Maps will tell you you need to be in the second from right lane on a turn, etc.) and this level of localization will allow for autonomy. Furthermore, GPS fails in many urban areas where street view localization excels. Large structures can block GPS signals (e.g. tall buildings) but these structures can aid localization with our solution.


### Why should people (everyone) care about the problem?
Because the price of the sensor kit of current autonomous vehicle implementations are extremely expensive, upwards of $100,000 per vehicle. Companies such as Tesla and comma.ai have demonstrated that perception capabilities can be implemented on a limited camera-based sensor suite, and we hope that our solution can help enhance the localization of these types of vehicles. This will bring down the overall cost of such vehicles and allows them to be accessible to a much broader audience. 
### What has been done so far to address this problem?
The problem of localization with Google Street View has been tackled before in several other research papers. Our goal is to combine additional low-cost sensors to enhance these methods. 
## Specific project scope

### What subset of the overall big picture problem are you addressing in particular?
We are combining localization via Street View with new sensors to improve the accuracy of this approach.
### How does solving this subproblem lead towards solving the big picture problem?
If we can add additional cost efficient sensors we can get closer to the goal of a cheap sensor suite that makes autonomous driving cars more affordable.
### What is your specific approach to solving this subproblem?
We are going to be adding IMU data to our video stream for a more comprehensive view of our environment. 
### How can you be reasonably sure this approach will result in a solution?
We are taking traditional GPS and using previously published methods that give relatively accurate localization to provide additional context to our environment and, in turn, improve accuracy.
### How will we know that this subproblem has been satisfactorily solved, using quantitative metrics?
We will peform two methods of validation. First, we will capture camera/IMU data in a multiple locations for which we know the exact latitude and longitude, and compare our results with previous methods and with the phone GPS. Second, we will drive down Westwood Boulevard in a straight line, and evaluate the linearity of the localized coordinates and drift within the lane.
## Broader impact
(even if someone doesn't care about the big picture problem that you started with, why should they still care about the specific work that you've produced?  Who else can use your processes and results, and how?)

### What is the value of your approach beyond this specific solution?
If we are successful with our approach it would prove that it is possible to create an autonomous driving car much cheaper than those on the market today. A single phone can be used to localize the vehicle and can theoretically be placed on any car on the road.
### What is the value of this solution beyond solely solving this subproblem and getting us closer to solving the big picture problem?
With cheaper autonomous vehicles, they become more accessible to those who need them such as the elderly and disabled persons.
## Background / related work / references
Link to your literature review in your repo.

## System capabilities, validation deliverables, engineering tasks

### Concrete external deadlines (paper submissions):
Include dates as well as target proposed title / abstract for expected submission

### Detailed schedule (weekly capabilities / deliverables / tasks):
Link to schedule in your repo.
