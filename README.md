<pre>
_____/\\\\\\\\\\\____/\\\________/\\\__/\\\\\_____/\\\__/\\\\\\\\\\\\\\\__/\\\\\\\\\\\__/\\\\____________/\\\\__/\\\\\\\\\\\\\\\_        
 ___/\\\/////////\\\_\/\\\_______\/\\\_\/\\\\\\___\/\\\_\///////\\\/////__\/////\\\///__\/\\\\\\________/\\\\\\_\/\\\///////////__       
  __\//\\\______\///__\/\\\_______\/\\\_\/\\\/\\\__\/\\\_______\/\\\___________\/\\\_____\/\\\//\\\____/\\\//\\\_\/\\\_____________      
   ___\////\\\_________\/\\\_______\/\\\_\/\\\//\\\_\/\\\_______\/\\\___________\/\\\_____\/\\\\///\\\/\\\/_\/\\\_\/\\\\\\\\\\\_____     
    ______\////\\\______\/\\\_______\/\\\_\/\\\\//\\\\/\\\_______\/\\\___________\/\\\_____\/\\\__\///\\\/___\/\\\_\/\\\///////______    
     _________\////\\\___\/\\\_______\/\\\_\/\\\_\//\\\/\\\_______\/\\\___________\/\\\_____\/\\\____\///_____\/\\\_\/\\\_____________   
      __/\\\______\//\\\__\//\\\______/\\\__\/\\\__\//\\\\\\_______\/\\\___________\/\\\_____\/\\\_____________\/\\\_\/\\\_____________  
       _\///\\\\\\\\\\\/____\///\\\\\\\\\/___\/\\\___\//\\\\\_______\/\\\________/\\\\\\\\\\\_\/\\\_____________\/\\\_\/\\\\\\\\\\\\\\\_ 
        ___\///////////________\/////////_____\///_____\/////________\///________\///////////__\///______________\///__\///////////////__
</pre>

# Suntime

This application attempts to correctly guess the time and date (or a close estimation) of when an image was taken based on the position of the sun, length of shadows, and the shadow's angle relative to the object that casted it. 

It was originally theorized in 2022, and since numerous whitepapers have been written about performing such a task, but an actual implementation of which has still not been found.

## Features

- Shadow recognition and measurement
- Astronomical positioning of the sun.
- Object Detection

## Current Status

There have been about three attempts at implementation, and revision to the methods to acquire a correct date. Currently, the repository is little more than a developer playing around with computational vision. So, no need to even bother with it. 

## Installation

```bash
git clone https://github.com/yourusername/suntime.git
cd suntime
# Add installation instructions here (e.g., pip install .)
```

## Usage

```python
from suntime import Sun

sun = Sun(latitude, longitude)
sunrise = sun.get_sunrise_time()
sunset = sun.get_sunset_time()
print(f"Sunrise: {sunrise}, Sunset: {sunset}")
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Source of solar calculations, if any]
- [Any libraries or contributors]
