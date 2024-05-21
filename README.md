# sign-detection

road sign image classifier for ros2 turtlebot (SomSem. 2024 Robotics Project)
## Data
- Data2
  - only turtlebot nr 60
  - none: other signs and empty room
- data
    - turtlebot nr 60, 80, 30
    - in different lighting situations
## Models
- model_0
  - total params: 6,215 (24.28 KB)
  - validation accuracy: ~97%

## Execution 

Training:
```
cd src && python3 train.py
```

Testing:
```
cd src && python3 test.py
```
