import React, { useState, useEffect } from "react";
import './App.css';
import twitterLogo from "./assets/twitter-logo.svg";
import * as fcl from "@onflow/fcl";
import * as types from "@onflow/types";
import { mintNFT } from "./cadence/transactions/mintNFT_tx";
import { getTotalSupply } from "./cadence/scripts/getTotalSupply_script";
import { getIDs } from "./cadence/scripts/getID_script";
import { getMetadata } from "./cadence/scripts/getMetadata_script";

const TWITTER_HANDLE = "apshirokov";
const TWITTER_LINK = `https://twitter.com/${TWITTER_HANDLE}`;


fcl.config({
  "flow.network": "testnet",
  "app.detail.title": "GenlockFinal", // Change the title!
  "accessNode.api": "https://rest-testnet.onflow.org",
  "app.detail.icon": "https://storage.googleapis.com/nft-game-assets/logo.png",
  "discovery.wallet": "https://fcl-discovery.onflow.org/testnet/authn",
});

function App() {

  const [ user, setUser ] = useState();
  const [ images, setImages ] = useState([])

  const logIn = () => {
    fcl.authenticate();
  };

  const RenderGif = () => {
    const gifUrl = user?.addr
        ? "https://media.giphy.com/media/VbnUQpnihPSIgIXuZv/giphy-downsized.gif"
        : "https://i.giphy.com/media/Y2ZUWLrTy63j9T6qrK/giphy.webp";
    return <img className="gif-image" src={gifUrl} height="300px" alt="Funny gif"/>;
  };

  const RenderLogin = () => {
    return (
      <div>
        <button className="cta-button button-glow" onClick={() => logIn()}>
          Take it!
        </button>
      </div>
    );
  };

  const RenderMint = () => {
    return (
      <div>
        <div className="button-container">
          <button className="cta-button button-glow" onClick={() => mint()}>
            Mint
          </button>
        </div>
        {images.length > 0 ? 
          <>
            <h2>Your NFTs</h2>
              <div className="image-container">
                {images}
              </ div>
          </> 
        : ""}
    </div>
    );
  }

  const mint = async() => {

    let _totalSupply;
    try {
      _totalSupply = await fcl.query({
        cadence: `${getTotalSupply}`
      })
    } catch(err) {console.log(err)}
    
    const _id = parseInt(_totalSupply) + 1;
    
    try {
      const transactionId = await fcl.mutate({
        cadence: `${mintNFT}`,
        args: (arg, t) => [
          arg(user.addr, types.Address), //address to which NFT should be minted
          arg("Genlock # "+_id.toString(), types.String),
          arg("Shield", types.String),
          arg("https://storage.googleapis.com/nft-game-assets/images/"+_id+".png", types.String),
          // arg("https://storage.googleapis.com/nft-game-assets/images/10.png", types.String),
          arg("Shield", types.String),
        ],
        proposer: fcl.currentUser,
        payer: fcl.currentUser,
        limit: 99
      })
      console.log("Minting NFT now with transaction ID", transactionId);
      const transaction = await fcl.tx(transactionId).onceSealed();
      console.log("Testnet explorer link:", `https://testnet.flowscan.org/transaction/${transactionId}`);
      console.log(transaction);

      window.location.href = "flow://mylink?https%3A%2F%2Fstorage.googleapis.com%2Fnft-game-assets%2Fimages%2F11.png";
      // alert("NFT minted successfully!")
    } catch (error) {
      console.log(error);
      alert("Error minting NFT, please check the console for error details!")
    }
  }

  useEffect(() => {
    fcl.currentUser().subscribe(setUser);  
  }, [])

  useEffect(() => {
    if (user && user.addr) {
      mint();
    }
  }
  , [user]);

  return (
    <div className="App">
      <div className="container">
        <div className="header-container">
          <div className="logo-container">
            <img src="./logo.png" className="flow-logo" alt="flow logo"/>
            <p className="header">✨ Congradulations! ✨</p>
          </div>
          <RenderGif />
          {user && user.addr ? 
              <p className="sub-text"> Generating NFT... </p> : 
              <p className="sub-text"> Generate and mint your NFT </p> }
        </div>
        
        {user && user.addr ? <></> : <RenderLogin />}
      
      </div>
    </div>
  );
}

export default App;