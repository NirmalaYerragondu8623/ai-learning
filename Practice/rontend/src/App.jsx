import { useState, useEffect, useRef } from "react";

export default function App(){
  const [input,setInput]=useState("")
  const [submitted,setSubmitted]=useState("")
  const [messages,setMessages]=useState([])
  const [loading,setLoading]=useState(false)
  const bottomRef=useRef(null)

  const addMessage=async ()=>{
    if (!input.trim()) return
    setLoading(true)
    setTimeout(()=>setLoading(false),2000)

    setMessages(prev=>[...prev,{text:input,from:"user"}])
    setInput("")

    setLoading(true)

    const res=await fetch("http://localhost:8002/echo",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify({message:input,history:messages})
    })

    const data=await res.json()

    setMessages(prev=>[...prev,{text:data.reply,from:"assistant"}])

    setLoading(false)
  }

  useEffect(()=>{
    if (loading){
      document.title="Thinking.."
    }else{
      document.title=`Chat (${messages.length} messages)`
    }
  },[loading,messages])

  useEffect(()=>{
    bottomRef.current?.scrollIntoView({behavior:"smooth"})
  },[messages])

  return(
    <>
    <div>
      <input 
      value={input} 
      onChange={e=>setInput(e.target.value)}
      placeholder="Type Something"
      />
      <button onClick={addMessage}>Send</button>
      {messages.map((msg,index)=>(
        <p key={index}>{msg.from}: {msg.text}</p>
      ))}
    </div>
    <div ref={bottomRef}/> 
    </>   
  )
}