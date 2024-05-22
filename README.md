# sign-detection

road sign image classifier for ros2 turtlebot (SomSem. 2024 Robotics Project)
## Data
- Data2
  - only turtlebot nr 60
  - none: other signs and empty room
  - (45, 40)
- data
    - turtlebot nr 60, 80, 30
    - in different lighting situations
    - (96, 80)
## Models
- model_0
  - total params: 6,215 (24.28 KB)
  - validation accuracy: ~97%

- model4
  - without none
  - using train2

- model5
  - with none
  - using train2

## Execution 

Training:
```
cd src && python3 train.py
```

Testing:
```
cd src && python3 test.py
```
