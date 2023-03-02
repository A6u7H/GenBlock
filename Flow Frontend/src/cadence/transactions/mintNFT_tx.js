export const mintNFT = 
`
// REPLACE THIS WITH YOUR CONTRACT NAME + ADDRESS
import GenlockFinal from 0xd5c962029b823bf5 

// Do not change these
import NonFungibleToken from 0x631e88ae7f1d7c20
import MetadataViews from 0x631e88ae7f1d7c20

transaction(
  recipient: Address,
  name: String,
  description: String,
  thumbnail: String,
  category: String,
) {
  prepare(signer: AuthAccount) {
    if signer.borrow<&GenlockFinal.Collection>(from: GenlockFinal.CollectionStoragePath) != nil {
      return
    }

    // Create a new empty collection
    let collection <- GenlockFinal.createEmptyCollection()

    // save it to the account
    signer.save(<-collection, to: GenlockFinal.CollectionStoragePath)

    // create a public capability for the collection
    signer.link<&{NonFungibleToken.CollectionPublic, MetadataViews.ResolverCollection}>(
      GenlockFinal.CollectionPublicPath,
      target: GenlockFinal.CollectionStoragePath
    )
  }


  execute {
    // Borrow the recipient's public NFT collection reference
    let receiver = getAccount(recipient)
      .getCapability(GenlockFinal.CollectionPublicPath)
      .borrow<&{NonFungibleToken.CollectionPublic}>()
      ?? panic("Could not get receiver reference to the NFT Collection")

    // Mint the NFT and deposit it to the recipient's collection
    GenlockFinal.mintNFT(
      recipient: receiver,
      name: name,
      description: description,
      thumbnail: thumbnail,
      category: category,
    )
    
    log("Minted an NFT and stored it into the collection")
  } 
}
`