import React, { useEffect, useRef, useState } from 'react';
import { withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';
import Paper from '@material-ui/core/Paper';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableContainer from '@material-ui/core/TableContainer';
import TableHead from '@material-ui/core/TableHead';
import TableRow from '@material-ui/core/TableRow';
import Grid from '@material-ui/core/Grid';
import TextField from '@material-ui/core/TextField';
import Button from '@material-ui/core/Button';
import Select from '@material-ui/core/Select';
import MenuItem from '@material-ui/core/MenuItem';
import FormControl from '@material-ui/core/FormControl';
import InputLabel from '@material-ui/core/InputLabel';


import DataService from "../../services/DataService";
import styles, { StyledTableCell }  from './styles';




const ImageMatching = (props) => {
    const { classes } = props;

    console.log("================================== ImageMatching ======================================");


    const inputFile = useRef(null);

    // Component States
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [dogStatus, setDogStatus] = useState('');
    const [name, setName] = useState('');
    const [email, setEmail] = useState('');
    const [phone, setPhone] = useState('');
    const [selectedImage, setSelectedImage] = useState(null);


    // Setup Component
    useEffect(() => {

    }, []);

    // Handlers
    const handleImageUploadClick = () => {
        inputFile.current.click();
    };

    const handleOnChange = (event) => {
        const image = event.target.files[0];
        if (image) {
            setImage(URL.createObjectURL(image));
            setSelectedImage(image);
        }
    };

    
    const handleSubmit = (event) => {
        event.preventDefault();
        setPrediction(null);
    
        const formData = new FormData();
        formData.append("image", selectedImage);
        formData.append("dog_status", dogStatus);
        formData.append("name", name);
        formData.append("email", email);
        formData.append("phone", phone);


        DataService.ImageMatching(formData)
            .then(function (response) {
                console.log("API response:", response.data);
                if (response.data.Response === "No Matching Dogs Found" || Object.keys(response.data).length === 0) {
                    setPrediction({ matches: [], noMatches: true });
                } else {
                    const matchesArray = Object.values(response.data);
                    setPrediction({ matches: matchesArray, noMatches: false });
                }
            })
            .catch(error => {
                console.error("Error fetching data:", error);
                setPrediction({ matches: [], noMatches: true });
            });
};


    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxWidth="x1" className={classes.container}>
                    <Typography variant="h5" gutterBottom>Search For A Lost or Found Dog</Typography>
                    <Typography variant="b5" gutterBottom>
                        Select whether you have lost or found a dog, enter your contact info, select a picture of the dog and click submit to see if there are any matches. The picture and your contact info will be saved to be used for future searches. Please use pictures with only 1 dog.
                        </Typography>
                    <Divider />
                    
                    <Grid container spacing={2}>
                        <Grid item xs={12} md={4} style={{ paddingTop: '30px' }}> 
                            <form onSubmit={handleSubmit} style={{ maxWidth: '400px', margin: '0 auto' }}>
                                <Grid container spacing={2}>
                                <Grid item xs={12}>
                                    <FormControl fullWidth>
                                        <InputLabel>Dog Lost or Found?</InputLabel>
                                        <Select
                                            value={dogStatus}
                                            onChange={(e) => setDogStatus(e.target.value)}
                                            label="Dog Lost or Found?"
                                        >
                                            <MenuItem value="lost">Lost</MenuItem>
                                            <MenuItem value="found">Found</MenuItem>
                                        </Select>
                                    </FormControl>
                                </Grid>

                                    <Grid item xs={12}>
                                        <TextField
                                            fullWidth
                                            margin="normal"
                                            type="text"
                                            value={name}
                                            onChange={(e) => setName(e.target.value)}
                                            label="Your Name"
                                            variant="outlined"
                                         />
                                    </Grid>
                                    <Grid item xs={12}>
                                        <TextField
                                            fullWidth
                                            margin="normal"
                                            type="email" 
                                            value={email}
                                            onChange={(e) => setEmail(e.target.value)}
                                            label="Your Email Address"
                                            variant="outlined"
                                        />
                                    </Grid>

                                    <Grid item xs={12}>
                                        <TextField
                                            fullWidth
                                            margin="normal"
                                            type="tel"
                                            value={phone}
                                            onChange={(e) => setPhone(e.target.value)}
                                            label="Your Phone Number"
                                            variant="outlined"
                                        />
                                    </Grid>

                                    <Grid item xs={12}>
                                        <Button 
                                            type="submit" 
                                            variant="contained" 
                                            color="primary"
                                            style={{ margin: '10px 0' }}
                                        >
                                            Submit
                                        </Button>
                                    </Grid>
                                </Grid>
                        
                            </form>
                        </Grid>

                        <Grid item xs={12} md={8}> 
                            <div className={classes.dropzone} onClick={() => handleImageUploadClick()}>
                                <input
                                    type="file"
                                    accept="image/*"
                                    capture="camera"
                                    on
                                    autocomplete="off"
                                    tabindex="-1"
                                    className={classes.fileInput}
                                    ref={inputFile}
                                    onChange={(event) => handleOnChange(event)}
                                    />
                                <div>
                                    <img className={classes.preview} src={image} alt="Preview"/>
                                </div>

                                <div className={classes.help}>Click to upload a picture
                                </div>
                            </div>
                        </Grid>
                    </Grid>
                
                    
                    {
                    prediction ?
                        (prediction.noMatches ?
                            <Typography variant="h6" align="center" style={{color: 'red'}}> 
                                No Matching Dogs Found 
                            </Typography>
                            :
                            <TableContainer component={Paper}>
                                <Table>
                                    <TableHead>
                                        <TableRow>
                                            <StyledTableCell align="center">Image</StyledTableCell>
                                            <StyledTableCell align="center">Similarity</StyledTableCell>
                                            <StyledTableCell align="center">Name</StyledTableCell>
                                            <StyledTableCell align="center">Phone Number</StyledTableCell>
                                            <StyledTableCell align="center">Email</StyledTableCell>
                                        </TableRow>
                                     </TableHead>

                                     <TableBody>
                                        {prediction.matches.map((match, index) => (
                                            <TableRow key={index}>
                                                <TableCell align="center">
                                                    <img src={`data:image/jpeg;base64,${match.image}`} alt="Matched Dog" style={{ width: '100px', height: 'auto' }} />
                                                </TableCell>
                                                <TableCell align="center" style={{ fontWeight: 'bold' }}>{match.Similarity}</TableCell>
                                                <TableCell align="center" style={{ fontWeight: 'bold' }}>{match.Name}</TableCell>
                                                <TableCell align="center" style={{ fontWeight: 'bold' }}>{match['Phone Number']}</TableCell>
                                                <TableCell align="center" style={{ fontWeight: 'bold' }}>{match.Email}</TableCell>
                                            </TableRow>
                                                   ))}
                                    </TableBody>
                                </Table>
                            </TableContainer>
                        )
                        : null
                       
}

                </Container>
            </main>
        </div>
    );
};

export default withStyles(styles)(ImageMatching);